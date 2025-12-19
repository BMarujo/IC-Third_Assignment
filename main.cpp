#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <zstd.h>
#include <stdexcept>

// Configuration
constexpr size_t CHUNK_SIZE = 32 * 1024 * 1024; // 32MB chunks
constexpr int DEFAULT_COMPRESSION_LEVEL = 3;    // Zstd default is usually 3

// --- Helper Utilities ---

class Timer {
    using Clock = std::chrono::high_resolution_clock;
    std::chrono::time_point<Clock> start_time;
public:
    Timer() : start_time(Clock::now()) {}
    double elapsed() {
        return std::chrono::duration<double>(Clock::now() - start_time).count();
    }
};

// Get file size for progress display
uint64_t get_file_size(std::ifstream& file) {
    std::streampos current = file.tellg();
    file.seekg(0, std::ios::end);
    std::streampos end = file.tellg();
    file.seekg(current, std::ios::beg);
    return static_cast<uint64_t>(end);
}

void print_progress(uint64_t processed, uint64_t total) {
    if (total == 0) return;
    int width = 50;
    float progress = (float)processed / total;
    int pos = (int)(width * progress);
    
    std::cout << "\r[";
    for (int i = 0; i < width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "% " << std::flush;
}

// --- Binary I/O Helpers ---

void write_uint64(std::ofstream& out, uint64_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

bool read_uint64(std::ifstream& in, uint64_t& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    return in.gcount() == sizeof(value);
}

// --- Shuffle Logic (BF16 Optimization) ---

void shuffle_bf16(const uint8_t* src, uint8_t* dst, size_t size) {
    if (size % 2 != 0) throw std::runtime_error("Data size must be even for BF16 shuffle");
    size_t half = size / 2;
    for (size_t i = 0; i < half; ++i) {
        dst[i] = src[2 * i + 1];      // High Byte
        dst[half + i] = src[2 * i];   // Low Byte
    }
}

void unshuffle_bf16(const uint8_t* src, uint8_t* dst, size_t size) {
    if (size % 2 != 0) throw std::runtime_error("Data size must be even for BF16 unshuffle");
    size_t half = size / 2;
    for (size_t i = 0; i < half; ++i) {
        dst[2 * i + 1] = src[i];      // High Byte
        dst[2 * i] = src[half + i];   // Low Byte
    }
}

// --- Core Operations ---

void compress(const std::string& input_path, const std::string& output_path, int level) {
    std::ifstream input(input_path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open input: " + input_path);
    std::ofstream output(output_path, std::ios::binary);
    if (!output) throw std::runtime_error("Cannot open output: " + output_path);

    uint64_t total_input_size = get_file_size(input);
    std::cout << "Compressing " << input_path << " (Level " << level << ")" << std::endl;

    // 1. Handle Header
    // We assume the file starts with a uint64_t indicating header size, followed by header data.
    uint64_t header_size = 0;
    if (!read_uint64(input, header_size)) throw std::runtime_error("File too small for header size");

    std::vector<uint8_t> header(header_size);
    input.read(reinterpret_cast<char*>(header.data()), header_size);
    if (input.gcount() != static_cast<std::streamsize>(header_size)) throw std::runtime_error("Header truncated");

    // Write Header (Uncompressed) to allow easy inspection later
    write_uint64(output, header_size);
    output.write(reinterpret_cast<const char*>(header.data()), header_size);

    // 2. Process Data Chunks
    std::vector<uint8_t> raw_buf(CHUNK_SIZE);
    std::vector<uint8_t> shuffled_buf(CHUNK_SIZE);
    std::vector<uint8_t> comp_buf; // Size will be set by compressBound

    size_t processed_bytes = sizeof(header_size) + header_size;
    uint64_t total_out_size = processed_bytes;
    
    Timer timer;

    while (true) {
        input.read(reinterpret_cast<char*>(raw_buf.data()), CHUNK_SIZE);
        size_t bytes_read = input.gcount();
        if (bytes_read == 0) break;

        // Shuffle
        shuffle_bf16(raw_buf.data(), shuffled_buf.data(), bytes_read);

        // Compress
        size_t bound = ZSTD_compressBound(bytes_read);
        if (comp_buf.size() < bound) comp_buf.resize(bound);

        size_t c_size = ZSTD_compress(comp_buf.data(), bound, shuffled_buf.data(), bytes_read, level);
        if (ZSTD_isError(c_size)) throw std::runtime_error(ZSTD_getErrorName(c_size));

        // Write Chunk Format: [Raw Size (u64)] [Compressed Size (u64)] [Data...]
        write_uint64(output, bytes_read);
        write_uint64(output, c_size);
        output.write(reinterpret_cast<const char*>(comp_buf.data()), c_size);

        processed_bytes += bytes_read;
        total_out_size += (sizeof(uint64_t) * 2) + c_size;
        
        print_progress(processed_bytes, total_input_size);
    }
    
    std::cout << "\nDone in " << timer.elapsed() << "s" << std::endl;
    std::cout << "Ratio: " << std::fixed << std::setprecision(2) 
              << (double)processed_bytes / total_out_size << "x (" 
              << processed_bytes << " -> " << total_out_size << " bytes)" << std::endl;
}

void decompress(const std::string& input_path, const std::string& output_path) {
    std::ifstream input(input_path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open input: " + input_path);
    std::ofstream output(output_path, std::ios::binary);
    if (!output) throw std::runtime_error("Cannot open output: " + output_path);

    uint64_t total_input_size = get_file_size(input);
    std::cout << "Decompressing " << input_path << "..." << std::endl;

    // 1. Recover Header
    uint64_t header_size = 0;
    if (!read_uint64(input, header_size)) throw std::runtime_error("Missing header size");

    std::vector<uint8_t> header(header_size);
    input.read(reinterpret_cast<char*>(header.data()), header_size);
    
    // Write Header
    write_uint64(output, header_size);
    output.write(reinterpret_cast<const char*>(header.data()), header_size);

    // 2. Decompress Chunks
    std::vector<uint8_t> comp_buf;
    std::vector<uint8_t> shuffled_buf;
    std::vector<uint8_t> final_buf;
    
    ZSTD_DCtx* dctx = ZSTD_createDCtx();
    uint64_t chunk_raw_size = 0;
    uint64_t chunk_comp_size = 0;
    
    Timer timer;

    while (read_uint64(input, chunk_raw_size)) {
        if (!read_uint64(input, chunk_comp_size)) throw std::runtime_error("Corrupted chunk header");

        if (comp_buf.size() < chunk_comp_size) comp_buf.resize(chunk_comp_size);
        input.read(reinterpret_cast<char*>(comp_buf.data()), chunk_comp_size);

        if (shuffled_buf.size() < chunk_raw_size) shuffled_buf.resize(chunk_raw_size);
        if (final_buf.size() < chunk_raw_size) final_buf.resize(chunk_raw_size);

        size_t d_size = ZSTD_decompressDCtx(dctx, shuffled_buf.data(), chunk_raw_size, comp_buf.data(), chunk_comp_size);
        if (ZSTD_isError(d_size)) throw std::runtime_error(ZSTD_getErrorName(d_size));

        unshuffle_bf16(shuffled_buf.data(), final_buf.data(), chunk_raw_size);
        output.write(reinterpret_cast<const char*>(final_buf.data()), chunk_raw_size);
        
        print_progress(input.tellg(), total_input_size);
    }

    ZSTD_freeDCtx(dctx);
    std::cout << "\nDone in " << timer.elapsed() << "s" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <compress|decompress> <input> <output> [level 1-22]" << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    std::string input = argv[2];
    std::string output = argv[3];
    int level = (argc >= 5) ? std::stoi(argv[4]) : DEFAULT_COMPRESSION_LEVEL;

    try {
        if (mode == "compress") compress(input, output, level);
        else if (mode == "decompress") decompress(input, output);
        else std::cerr << "Unknown mode: " << mode << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}