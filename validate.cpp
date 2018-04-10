/*
 * Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in 
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <stdlib.h>
#include <memory.h>
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <linux/pci.h>

#include <sys/mman.h>

using namespace std;

#include "gdrapi.h"
#include "common.hpp"



#define _XOPEN_SOURCE 700
#include <fcntl.h> /* open */
#include <stdint.h> /* uint64_t  */
#include <stdio.h> /* printf */
#include <stdlib.h> /* size_t */
#include <unistd.h> /* pread, sysconf */

typedef struct {
	uint64_t pfn : 54;
	unsigned int soft_dirty : 1;
	unsigned int file_page : 1;
	unsigned int swapped : 1;
	unsigned int present : 1;
} PagemapEntry;

/* Parse the pagemap entry for the given virtual address.
 *
 * @param[out] entry      the parsed entry
 * @param[in]  pagemap_fd file descriptor to an open /proc/pid/pagemap file
 * @param[in]  vaddr      virtual address to get entry for
 * @return 0 for success, 1 for failure
 */
int pagemap_get_entry(PagemapEntry *entry, int pagemap_fd, uintptr_t vaddr)
{
	size_t nread;
	ssize_t ret;
	uint64_t data;

	nread = 0;
	while (nread < sizeof(data)) {
		ret = pread(pagemap_fd, &data, sizeof(data),
		        (vaddr / sysconf(_SC_PAGE_SIZE)) * sizeof(data) + nread);
		nread += ret;
		if (ret <= 0) {
			return 1;
		}
	}
	entry->pfn = data & (((uint64_t)1 << 54) - 1);
	entry->soft_dirty = (data >> 54) & 1;
	entry->file_page = (data >> 61) & 1;
	entry->swapped = (data >> 62) & 1;
	entry->present = (data >> 63) & 1;
	return 0;
}

/* Convert the given virtual address to physical using /proc/PID/pagemap.
 *
 * @param[out] paddr physical address
 * @param[in]  pid   process to convert for
 * @param[in] vaddr virtual address to get entry for
 * @return 0 for success, 1 for failure
 */
int virt_to_phys_user(uintptr_t *paddr, pid_t pid, uintptr_t vaddr)
{
	char pagemap_file[BUFSIZ];
	int pagemap_fd;

	snprintf(pagemap_file, sizeof(pagemap_file), "/proc/%ju/pagemap", (uintmax_t)pid);
	pagemap_fd = open(pagemap_file, O_RDONLY);
	if (pagemap_fd < 0) {
		return 1;
	}
	PagemapEntry entry;
	if (pagemap_get_entry(&entry, pagemap_fd, vaddr)) {
		return 1;
	}
	close(pagemap_fd);
	*paddr = (entry.pfn * sysconf(_SC_PAGE_SIZE)) + (vaddr % sysconf(_SC_PAGE_SIZE));
	return 0;
}


#define MAP_SIZE 4096UL
#define MAP_MASK (MAP_SIZE - 1)


int main(int argc, char *argv[])
{
    void *dummy;
    ASSERTRT(cudaMalloc(&dummy, 0));

    const size_t _size = 256*1024+16; //32*1024+8;
    const size_t size = (_size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;

    printf("buffer size: %zu\n", size);
    CUdeviceptr d_A;
	CUdeviceptr d_B;

	ASSERTDRV(cuMemAlloc(&d_A, size));
	ASSERTDRV(cuMemAlloc(&d_B, 2*size));

	ASSERTDRV(cuMemsetD8(d_A, 0xA5, size));
	ASSERTDRV(cuMemsetD8(d_B, 0xA5, 2*size));
	//OUT << "device ptr: " << hex << d_A << dec << endl;

    unsigned int flag = 1;
    ASSERTDRV(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, d_A));
	ASSERTDRV(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, d_B));

    uint32_t *init_buf = new uint32_t[size];
    uint32_t *copy_buf = new uint32_t[size];

    init_hbuf_walking_bit(init_buf, size);
    memset(copy_buf, 0xA5, size * sizeof(*copy_buf));

    gdr_t g = gdr_open();
    ASSERT_NEQ(g, (void*)0);

    gdr_mh_t mh;
	gdr_mh_t mh2;
	BEGIN_CHECK {
        CUdeviceptr d_ptr = d_A;
		CUdeviceptr d_ptr2 = d_B;

		uint64_t addr[16];

        // tokens are optional in CUDA 6.0
        // wave out the test if GPUDirectRDMA is not enabled
		BREAK_IF_NEQ(gdr_pin_buffer(g, d_ptr, size, 0, 0, &mh), 0);
        ASSERT_NEQ(mh, 0U);
		if(gdr_map_dma(g, mh, 3, 0, 0, addr, 16) <= 0) {
			printf("couldn't get dma address\n");
			return 1;
		}

		int fd = open("/sys/bus/pci/devices/0000:01:00.0/resource1", O_RDWR | O_SYNC);
		if (fd < 0)
			return 1;

		const uintptr_t bar1_base = 0xc0000000;
		const uintptr_t dma_address = addr[0];
		if(dma_address < bar1_base) {
			printf("dma address (%p) below BAR1\n", dma_address);
			return 1;
		}

		const uintptr_t dma_offset = dma_address - bar1_base;

		/* Map one page */
		printf("mmap(%d, %lu, %#x, %#x, %d, %#lx)\n", 0,
		    MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED,
		    fd, dma_offset);

		auto map_base = reinterpret_cast<uintptr_t>(mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, dma_offset & ~MAP_MASK));
		printf("mapped dma addr to %p\n", map_base);

		BREAK_IF_NEQ(gdr_pin_buffer(g, d_ptr2, 2*size, 0, 0, &mh2), 0);
		ASSERT_NEQ(mh2, 0U);
		gdr_map_dma(g, mh2, 3, 0, 0, addr, 16);


        void *bar_ptr  = NULL;
        ASSERT_EQ(gdr_map(g, mh, &bar_ptr, size), 0);
        //OUT << "bar_ptr: " << bar_ptr << endl;

        gdr_info_t info;
        ASSERT_EQ(gdr_get_info(g, mh, &info), 0);
        int off = d_ptr - info.va;
        cout << "off: " << off << endl;

        uint32_t *buf_ptr = (uint32_t *)((char *)bar_ptr + off);
        //OUT << "buf_ptr:" << buf_ptr << endl;

        printf("check 1: MMIO CPU initialization + read back via cuMemcpy D->H\n");
        init_hbuf_walking_bit(buf_ptr, size);
        //mmiowcwb();
        ASSERTDRV(cuMemcpyDtoH(copy_buf, d_ptr, size));
        //ASSERTDRV(cuCtxSynchronize());
        compare_buf(init_buf, copy_buf, size);
        memset(copy_buf, 0xA5, size * sizeof(*copy_buf));

        printf("check 2: gdr_copy_to_bar() + read back via cuMemcpy D->H\n");
        gdr_copy_to_bar(buf_ptr, init_buf, size);
        ASSERTDRV(cuMemcpyDtoH(copy_buf, d_ptr, size));
        //ASSERTDRV(cuCtxSynchronize());
        compare_buf(init_buf, copy_buf, size);
        memset(copy_buf, 0xA5, size * sizeof(*copy_buf));

        printf("check 3: gdr_copy_to_bar() + read back via gdr_copy_from_bar()\n");
        gdr_copy_to_bar(buf_ptr, init_buf, size);
        gdr_copy_from_bar(copy_buf, buf_ptr, size);
        //ASSERTDRV(cuCtxSynchronize());
        compare_buf(init_buf, copy_buf, size);
        memset(copy_buf, 0xA5, size * sizeof(*copy_buf));

        int extra_dwords = 5;
        int extra_off = extra_dwords * sizeof(uint32_t);
        printf("check 4: gdr_copy_to_bar() + read back via gdr_copy_from_bar() + %d dwords offset\n", extra_dwords);
        gdr_copy_to_bar(buf_ptr + extra_dwords, init_buf, size - extra_off);
        gdr_copy_from_bar(copy_buf, buf_ptr + extra_dwords, size - extra_off);
        compare_buf(init_buf, copy_buf, size - extra_off);
        memset(copy_buf, 0xA5, size * sizeof(*copy_buf));

        extra_off = 11;
        printf("check 5: gdr_copy_to_bar() + read back via gdr_copy_from_bar() + %d bytes offset\n", extra_off);
        gdr_copy_to_bar((char*)buf_ptr + extra_off, init_buf, size);
        gdr_copy_from_bar(copy_buf, (char*)buf_ptr + extra_off, size);
        compare_buf(init_buf, copy_buf, size);

        printf("unampping\n");
        ASSERT_EQ(gdr_unmap(g, mh, bar_ptr, size), 0);
        printf("unpinning\n");
        ASSERT_EQ(gdr_unpin_buffer(g, mh), 0);
    } END_CHECK;
    ASSERT_EQ(gdr_close(g), 0);

    ASSERTDRV(cuMemFree(d_A));
    return 0;
}

/*
 * Local variables:
 *  c-indent-level: 4
 *  c-basic-offset: 4
 *  tab-width: 4
 *  indent-tabs-mode: nil
 * End:
 */
