OUTPUT_ARCH(riscv)
ENTRY(_start)

/* SRAM_SIZE is passed via --defsym=SRAM_SIZE=<bytes> from the Makefile */
MEMORY {
    SRAM (rwx)  : ORIGIN = 0x00000000, LENGTH = SRAM_SIZE
}

SECTIONS {
    /* Reset vector at 0x0; RESET_VECTOR=0x40 so _start lands at 0x40 */
    .vectors : {
        KEEP(*(.vectors))
        . = 0x40;
    } > SRAM

    .text : {
        *(.text.init)
        *(.text .text.*)
        *(.rodata .rodata.*)
        *(.srodata .srodata.*)
        . = ALIGN(4);
        _sidata = .;
    } > SRAM

    /* AT(_sidata): LMA == VMA for pure-SRAM builds (harmless no-op copy in crt0) */
    .data : AT(_sidata) {
        . = ALIGN(4);
        _sdata = .;
        PROVIDE(__global_pointer$ = . + 0x800);
        *(.data .data.*)
        *(.sdata .sdata.*)
        . = ALIGN(4);
        _edata = .;
    } > SRAM

    .bss : {
        . = ALIGN(4);
        _sbss = .;
        *(.bss .bss.*)
        *(.sbss .sbss.*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } > SRAM

    PROVIDE(_stack_top = ORIGIN(SRAM) + LENGTH(SRAM) - 16);

    /DISCARD/ : { *(.note.GNU-stack) *(.comment) }
}
