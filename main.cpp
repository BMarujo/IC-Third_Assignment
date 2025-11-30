#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <chrono>
#include <zstd.h>
#include <stdexcept>

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

// Function to read file into buffer
std::vector<uint8_t> read_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open file: " + filename);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(size);
    if (!file.read((char*)buffer.data(), size)) throw std::runtime_error("Read error");
    return buffer;
}

// Function to write buffer to file
void write_file(const std::string& filename, const std::vector<uint8_t>& buffer) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open output file: " + filename);
    file.write((const char*)buffer.data(), buffer.size());
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

void compress(const std::string& input_path, const std::string& output_path) {
    std::cout << "Loading " << input_path << "..." << std::endl;
    auto raw_data = read_file(input_path);
    
    if (raw_data.size() < 8) throw std::runtime_error("File too small");
    
    // Parse Header Size (Little Endian uint64)
    uint64_t header_size = 0;
    memcpy(&header_size, raw_data.data(), 8);
    
    std::cout << "Header Size: " << header_size << std::endl;
    
    size_t data_offset = 8 + header_size;
    if (raw_data.size() < data_offset) throw std::runtime_error("File smaller than header implies");
    
    size_t data_len = raw_data.size() - data_offset;
    std::cout << "Data Size: " << data_len << " bytes" << std::endl;
    
    std::vector<uint8_t> processed_data(raw_data.size());
    
    // Copy Header Size and Header as is
    memcpy(processed_data.data(), raw_data.data(), data_offset);
    
    // Shuffle Data
    std::cout << "Shuffling BF16 data..." << std::endl;
    Timer t_shuffle;
    shuffle_bf16(raw_data.data() + data_offset, processed_data.data() + data_offset, data_len);
    std::cout << "Shuffle time: " << t_shuffle.elapsed() << "s" << std::endl;
    
    // Compress
    std::cout << "Compressing with Zstd..." << std::endl;
    size_t max_dst_size = ZSTD_compressBound(processed_data.size());
    std::vector<uint8_t> compressed_data(max_dst_size);
    
    Timer t_compress;
    int compression_level = 15; 
    
    // Simple API
    size_t c_size = ZSTD_compress(compressed_data.data(), max_dst_size, 
                                  processed_data.data(), processed_data.size(), 
                                  compression_level);
    
    if (ZSTD_isError(c_size)) {
        throw std::runtime_error(std::string("Zstd error: ") + ZSTD_getErrorName(c_size));
    }
    
    compressed_data.resize(c_size);
    std::cout << "Compression time: " << t_compress.elapsed() << "s" << std::endl;
    std::cout << "Original size: " << raw_data.size() << std::endl;
    std::cout << "Compressed size: " << c_size << std::endl;
    std::cout << "Ratio: " << (double)raw_data.size() / c_size << "x" << std::endl;
    
    write_file(output_path, compressed_data);
    std::cout << "Saved to " << output_path << std::endl;
}

void decompress(const std::string& input_path, const std::string& output_path) {
    std::cout << "Loading " << input_path << "..." << std::endl;
    auto compressed_data = read_file(input_path);
    
    unsigned long long const r_size = ZSTD_getFrameContentSize(compressed_data.data(), compressed_data.size());
    if (r_size == ZSTD_CONTENTSIZE_ERROR) throw std::runtime_error("Not a zstd file");
    if (r_size == ZSTD_CONTENTSIZE_UNKNOWN) throw std::runtime_error("Unknown original size");
    
    std::vector<uint8_t> decompressed_data(r_size);
    
    std::cout << "Decompressing..." << std::endl;
    Timer t_decompress;
    
    ZSTD_DCtx* dctx = ZSTD_createDCtx();
    size_t d_size = ZSTD_decompressDCtx(dctx, decompressed_data.data(), r_size, 
                                        compressed_data.data(), compressed_data.size());
                                        
    if (ZSTD_isError(d_size)) {
        ZSTD_freeDCtx(dctx);
        throw std::runtime_error(std::string("Zstd error: ") + ZSTD_getErrorName(d_size));
    }
    ZSTD_freeDCtx(dctx);
    std::cout << "Decompression time: " << t_decompress.elapsed() << "s" << std::endl;
    
    uint64_t header_size = 0;
    memcpy(&header_size, decompressed_data.data(), 8);
    size_t data_offset = 8 + header_size;
    size_t data_len = d_size - data_offset;
    
    std::cout << "Unshuffling data..." << std::endl;
    Timer t_unshuffle;
    
    std::vector<uint8_t> final_data(d_size);
    
    // Copy header part
    memcpy(final_data.data(), decompressed_data.data(), data_offset);
    
    // Unshuffle data part
    unshuffle_bf16(decompressed_data.data() + data_offset, final_data.data() + data_offset, data_len);
    
    std::cout << "Unshuffle time: " << t_unshuffle.elapsed() << "s" << std::endl;
    
    write_file(output_path, final_data);
    std::cout << "Saved to " << output_path << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <compress|decompress> <input> <output>" << std::endl;
        return 1;
    }
    
    std::string mode = argv[1];
    std::string input = argv[2];
    std::string output = argv[3];
    
    try {
        if (mode == "compress") {
            compress(input, output);
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
