#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <chrono>
#include <cstdint>
#include <zstd.h>
#include <stdexcept>

constexpr size_t CHUNK_SIZE = 32 * 1024 * 1024; // 32MB blocks to limit memory

// Helper to measure time
class Timer {
    using Clock = std::chrono::high_resolution_clock;
    std::chrono::time_point<Clock> start_time;
public:
    Timer() : start_time(Clock::now()) {}
    double elapsed() {
        auto end_time = Clock::now();
        return std::chrono::duration<double>(end_time - start_time).count();
    }
};

void write_uint64(std::ofstream& out, uint64_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!out) throw std::runtime_error("Write error while writing uint64");
}

bool read_uint64(std::ifstream& in, uint64_t& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (in.gcount() == 0) return false; // EOF
    if (in.gcount() != static_cast<std::streamsize>(sizeof(value))) {
        throw std::runtime_error("Unexpected EOF while reading uint64");
    }
    return true;
}

// Shuffle function for BF16 (2 bytes)
// Separates MSBs and LSBs into two contiguous blocks
void shuffle_bf16(const uint8_t* src, uint8_t* dst, size_t size) {
    if (size % 2 != 0) throw std::runtime_error("Data size must be even for BF16 shuffle");
    size_t half = size / 2;
    for (size_t i = 0; i < half; ++i) {
        dst[i] = src[2 * i + 1];      // MSB
        dst[half + i] = src[2 * i];   // LSB
    }
}

// Unshuffle function
void unshuffle_bf16(const uint8_t* src, uint8_t* dst, size_t size) {
    if (size % 2 != 0) throw std::runtime_error("Data size must be even for BF16 unshuffle");
    size_t half = size / 2;
    for (size_t i = 0; i < half; ++i) {
        dst[2 * i + 1] = src[i];      // MSB
        dst[2 * i] = src[half + i];   // LSB
    }
}

void compress(const std::string& input_path, const std::string& output_path, int compression_level = 15) {
    std::cout << "Opening input and output streams..." << std::endl;
    std::ifstream input(input_path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open file: " + input_path);
    std::ofstream output(output_path, std::ios::binary);
    if (!output) throw std::runtime_error("Cannot open output file: " + output_path);

    uint64_t header_size = 0;
    input.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
    if (input.gcount() != static_cast<std::streamsize>(sizeof(header_size))) {
        throw std::runtime_error("File too small to contain header size");
    }

    std::vector<uint8_t> header(header_size);
    if (!input.read(reinterpret_cast<char*>(header.data()), header_size)) {
        throw std::runtime_error("Failed to read header");
    }

    std::cout << "Header Size: " << header_size << std::endl;
    
    // Write header size and header uncompressed so decompression can stream
    write_uint64(output, header_size);
    output.write(reinterpret_cast<const char*>(header.data()), header_size);
    if (!output) throw std::runtime_error("Failed to write header to output");

    std::vector<uint8_t> raw_chunk(CHUNK_SIZE);
    std::vector<uint8_t> shuffled_chunk(CHUNK_SIZE);
    std::vector<uint8_t> compressed_chunk;

    size_t total_in = 0;
    size_t total_out = sizeof(header_size) + header_size;
    // `compression_level` is provided by caller (default 15)

    std::cout << "Processing chunks of " << CHUNK_SIZE << " bytes..." << std::endl;
    Timer t_total;

    while (true) {
        input.read(reinterpret_cast<char*>(raw_chunk.data()), raw_chunk.size());
        std::streamsize bytes_read = input.gcount();
        if (bytes_read == 0) break;
        if (bytes_read % 2 != 0) throw std::runtime_error("Data size must be even for BF16 shuffle");

        uint64_t chunk_size = static_cast<uint64_t>(bytes_read);
        shuffled_chunk.resize(bytes_read);

        shuffle_bf16(raw_chunk.data(), shuffled_chunk.data(), bytes_read);

        size_t max_dst_size = ZSTD_compressBound(chunk_size);
        compressed_chunk.resize(max_dst_size);
        size_t c_size = ZSTD_compress(compressed_chunk.data(), max_dst_size,
                                      shuffled_chunk.data(), chunk_size,
                                      compression_level);

        if (ZSTD_isError(c_size)) {
            throw std::runtime_error(std::string("Zstd error: ") + ZSTD_getErrorName(c_size));
        }

        write_uint64(output, chunk_size);
        write_uint64(output, static_cast<uint64_t>(c_size));
        output.write(reinterpret_cast<const char*>(compressed_chunk.data()), c_size);
        if (!output) throw std::runtime_error("Failed to write compressed chunk");

        total_in += static_cast<size_t>(chunk_size);
        total_out += sizeof(uint64_t) * 2 + static_cast<size_t>(c_size);
    }

    std::cout << "Compression time: " << t_total.elapsed() << "s" << std::endl;
    std::cout << "Original size: " << total_in + sizeof(header_size) + header_size << std::endl;
    std::cout << "Compressed size: " << total_out << std::endl;
    if (total_out > 0) {
        std::cout << "Ratio: " << static_cast<double>(total_in + sizeof(header_size) + header_size) / total_out << "x" << std::endl;
    }
    std::cout << "Saved to " << output_path << std::endl;
}

void decompress(const std::string& input_path, const std::string& output_path) {
    std::cout << "Opening compressed input..." << std::endl;
    std::ifstream input(input_path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open file: " + input_path);
    std::ofstream output(output_path, std::ios::binary);
    if (!output) throw std::runtime_error("Cannot open output file: " + output_path);

    uint64_t header_size = 0;
    input.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
    if (input.gcount() != static_cast<std::streamsize>(sizeof(header_size))) {
        throw std::runtime_error("Corrupted compressed file: missing header size");
    }

    std::vector<uint8_t> header(header_size);
    if (!input.read(reinterpret_cast<char*>(header.data()), header_size)) {
        throw std::runtime_error("Corrupted compressed file: missing header data");
    }

    // Write header back to reconstructed file
    write_uint64(output, header_size);
    output.write(reinterpret_cast<const char*>(header.data()), header_size);
    if (!output) throw std::runtime_error("Failed to write header to output");

    std::vector<uint8_t> compressed_chunk;
    std::vector<uint8_t> shuffled_chunk;
    std::vector<uint8_t> final_chunk;

    size_t total_out = sizeof(header_size) + header_size;
    ZSTD_DCtx* dctx = ZSTD_createDCtx();
    if (!dctx) throw std::runtime_error("Failed to create ZSTD_DCtx");

    Timer t_decompress;
    while (true) {
        uint64_t chunk_size = 0;
        uint64_t compressed_size = 0;

        if (!read_uint64(input, chunk_size)) break; // EOF reached cleanly
        if (!read_uint64(input, compressed_size)) {
            ZSTD_freeDCtx(dctx);
            throw std::runtime_error("Corrupted compressed file: missing compressed size");
        }
        if (chunk_size % 2 != 0) {
            ZSTD_freeDCtx(dctx);
            throw std::runtime_error("Corrupted chunk: size not even for BF16");
        }

        compressed_chunk.resize(static_cast<size_t>(compressed_size));
        if (!input.read(reinterpret_cast<char*>(compressed_chunk.data()), compressed_size)) {
            ZSTD_freeDCtx(dctx);
            throw std::runtime_error("Corrupted compressed file: truncated chunk data");
        }

        shuffled_chunk.resize(static_cast<size_t>(chunk_size));
        size_t d_size = ZSTD_decompressDCtx(dctx,
                                            shuffled_chunk.data(), chunk_size,
                                            compressed_chunk.data(), compressed_size);
        if (ZSTD_isError(d_size)) {
            ZSTD_freeDCtx(dctx);
            throw std::runtime_error(std::string("Zstd error: ") + ZSTD_getErrorName(d_size));
        }
        if (d_size != chunk_size) {
            ZSTD_freeDCtx(dctx);
            throw std::runtime_error("Size mismatch after decompression");
        }

        final_chunk.resize(static_cast<size_t>(chunk_size));
        unshuffle_bf16(shuffled_chunk.data(), final_chunk.data(), chunk_size);

        output.write(reinterpret_cast<const char*>(final_chunk.data()), chunk_size);
        if (!output) {
            ZSTD_freeDCtx(dctx);
            throw std::runtime_error("Write error while writing decompressed chunk");
        }
        total_out += static_cast<size_t>(chunk_size);
    }

    ZSTD_freeDCtx(dctx);
    std::cout << "Decompression time: " << t_decompress.elapsed() << "s" << std::endl;
    std::cout << "Restored size: " << total_out << std::endl;
    std::cout << "Saved to " << output_path << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <compress|decompress> <input> <output> [compression_level]" << std::endl;
        return 1;
    }
    
    std::string mode = argv[1];
    std::string input = argv[2];
    std::string output = argv[3];
    int compression_level = 15;
    if (mode == "compress" && argc >= 5) {
        try {
            compression_level = std::stoi(argv[4]);
        } catch (const std::exception& e) {
            std::cerr << "Invalid compression level: " << argv[4] << std::endl;
            return 1;
        }
    }
    
    try {
        if (mode == "compress") {
            compress(input, output, compression_level);
        } else if (mode == "decompress") {
            decompress(input, output);
        } else {
            std::cerr << "Unknown mode: " << mode << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
