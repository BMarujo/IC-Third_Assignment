SHELL := /usr/bin/env bash

CXX ?= g++

# Common build flags
CPPFLAGS ?=
CXXFLAGS ?= -O3 -Wall -Wextra -std=c++17
LDFLAGS ?=
LDLIBS ?= -lzstd

# OpenMP flags for the parallel implementation
OMPFLAGS ?= -fopenmp

SERIAL_BIN := compressor
OMP_BIN := bf16_omp

SERIAL_SRC := main.cpp
OMP_SRC := bf16_omp.cpp

.PHONY: all serial omp clean help

all: serial omp

serial: $(SERIAL_BIN)

omp: $(OMP_BIN)

$(SERIAL_BIN): $(SERIAL_SRC)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $< -o $@ $(LDLIBS)

$(OMP_BIN): $(OMP_SRC)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OMPFLAGS) $(LDFLAGS) $< -o $@ $(LDLIBS)

clean:
	@rm -f $(SERIAL_BIN) $(OMP_BIN)
	@rm -f model.safetensors.zst model_restored.safetensors
	@rm -f model.safetensors.serial.zst model_restored.serial.safetensors
	@rm -f model.safetensors.omp.zst model_restored.omp.safetensors

help:
	@echo "Targets:"
	@echo "  all     Build both implementations (default)"
	@echo "  serial  Build serial implementation -> ./$(SERIAL_BIN)"
	@echo "  omp     Build OpenMP implementation -> ./$(OMP_BIN)"
	@echo "  clean   Remove binaries and common outputs"
