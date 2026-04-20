// spike_uart.cpp — Spike MMIO plugin for KWS-SoC UART (uart_mini)
//
// Simulates the UART peripheral at base address 0x40004000.
// Register layout matches uart_mini from KWS-SoC:
//   0x00  CSR    — control/status: bit[0] = enable (store: ignored)
//   0x04  DIV    — baud divisor (store: ignored on Spike)
//   0x08  FSTAT  — FIFO status: bit[8] = TX full (always returns 0 — never full)
//   0x0C  TX     — transmit data: store writes one character to host stdout
//
// Any store to the TX register is immediately printed to host stdout.
// No RX support needed for KWS firmware.

#include <riscv/devices.h>
#include <riscv/abstract_device.h>
#include <riscv/sim.h>
#include <cstdio>
#include <cstring>

class spike_uart_t : public abstract_device_t {
public:
    bool load(reg_t addr, size_t len, uint8_t* bytes) override {
        uint32_t val = 0;
        switch (addr) {
            case 0x00: val = 0x1;  break;  // CSR: enabled
            case 0x04: val = 0;    break;  // DIV
            case 0x08: val = 0;    break;  // FSTAT: TX never full
            case 0x0C: val = 0;    break;  // TX: read undefined
            default:   val = 0;    break;
        }
        memcpy(bytes, &val, len < 4 ? len : 4);
        return true;
    }

    bool store(reg_t addr, size_t len, const uint8_t* bytes) override {
        if (addr == 0x0C) {  // TX register
            putchar((char)bytes[0]);
            fflush(stdout);
        }
        // All other stores (CSR enable, DIV config) are silently accepted
        return true;
    }

    reg_t size() override { return 0x100; }
};

static spike_uart_t* spike_uart_parse(const void* fdt, const sim_t* sim,
                                       reg_t* base,
                                       const std::vector<std::string>& sargs) {
    *base = 0x40004000;
    return new spike_uart_t();
}

static std::string spike_uart_generate(const sim_t* sim,
                                        const std::vector<std::string>& sargs) {
    return "";  // no DTS entry needed
}

REGISTER_DEVICE(spike_uart, spike_uart_parse, spike_uart_generate)
