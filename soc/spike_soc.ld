/* spike_soc.ld — Linker script for bare-metal KWS firmware on Spike
 *
 * Spike loads ELF files with sections at their VMA. Spike's default RAM
 * starts at DRAM_BASE = 0x80000000. This script places everything there,
 * matching Spike's expected layout while keeping the same section structure
 * as the KWS-SoC.
 *
 * For SoC deployment (FPGA), relink using KWS-SoC's link.ld (XIP: code in
 * flash at 0x80000000, data in SRAM at 0x00000000) or sram_link.ld (all in
 * SRAM). The C source does not change between Spike and SoC builds.
 *
 * HTIF: Spike's host-target interface uses two symbols (tohost/fromhost) to
 * detect program exit. These are placed in their own section so the firmware
 * can optionally use them to signal completion.
 */

OUTPUT_ARCH(riscv)
ENTRY(_reset_vector)

MEMORY {
    RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 16M
}

SECTIONS {
    /* Reset vector — must be at the ELF entry point */
    .vectors : {
        KEEP(*(.vectors))
        . = 0x40;           /* _start lands at offset 0x40, matching SoC */
    } > RAM

    /* Code */
    .text : {
        *(.text.init)
        *(.text .text.*)
        . = ALIGN(4);
    } > RAM

    /* Read-only data (NNoM weights live here — const arrays) */
    .rodata : {
        *(.rodata .rodata.*)
        *(.srodata .srodata.*)
        . = ALIGN(4);
        _sidata = .;        /* LMA for .data (== VMA here, single region) */
    } > RAM

    /* Initialised data */
    .data : AT(_sidata) {
        . = ALIGN(4);
        _sdata = .;
        PROVIDE(__global_pointer$ = . + 0x800);
        *(.data .data.*)
        *(.sdata .sdata.*)
        . = ALIGN(4);
        _edata = .;
    } > RAM

    /* Zero-initialised data */
    .bss : {
        . = ALIGN(4);
        _sbss = .;
        *(.bss .bss.*)
        *(.sbss .sbss.*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } > RAM

    /* HTIF symbols for Spike exit signalling */
    .htif ALIGN(0x40) : {
        *(.htif)
    } > RAM

    _stack_top = ORIGIN(RAM) + LENGTH(RAM) - 16;

    /DISCARD/ : { *(.note.GNU-stack) *(.comment) *(.riscv.attributes) }
}
