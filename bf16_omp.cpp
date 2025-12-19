#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <stdexcept>
#include <omp.h>      // OpenMP Header
#include <zstd.h>     // Zstandard Header

// --- Configuration ---
constexpr size_t CHUNK_SIZE = 32 * 1024 * 1024; // 32MB per chunk
constexpr int BATCH_SIZE = 8;                   // Process 8 chunks at a time (approx 256MB RAM usage)
constexpr int DEFAULT_COMPRESSION_LEVEL = 3;

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
    if (progress > 1.0f) progress = 1.0f;
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

// --- BF16 Logic ---
void shuffle_bf16(const uint8_t* src, uint8_t* dst, size_t size) {
    size_t half = size / 2;
    for (size_t i = 0; i < half; ++i) {
        dst[i] = src[2 * i + 1];      // High Byte
        dst[half + i] = src[2 * i];   // Low Byte
    }
}

void unshuffle_bf16(const uint8_t* src, uint8_t* dst, size_t size) {
    size_t half = size / 2;
    for (size_t i = 0; i < half; ++i) {
        dst[2 * i + 1] = src[i];      // High Byte
        dst[2 * i] = src[half + i];   // Low Byte
    }
}

// --- Data Structure for Parallel Processing ---
struct Chunk {
    std::vector<uint8_t> raw_data;
    std::vector<uint8_t> comp_data;
    std::vector<uint8_t> scratch_buffer; // For shuffling
    uint64_t raw_size = 0;
    uint64_t comp_size = 0;
};

// --- Compression Implementation ---
void compress(const std::string& input_path, const std::string& output_path, int level) {
    std::ifstream input(input_path, std::ios::binary);
    std::ofstream output(output_path, std::ios::binary);
    if (!input || !output) throw std::runtime_error("File I/O error");

    uint64_t total_input_size = get_file_size(input);
    int num_threads = omp_get_max_threads();
    std::cout << "Compressing with " << num_threads << " threads (Batch size: " << BATCH_SIZE << ")..." << std::endl;

    // 1. Handle Header (Serial)
    uint64_t header_size = 0;
    if (!read_uint64(input, header_size)) throw std::runtime_error("Empty file or missing size");
    
    std::vector<uint8_t> header(header_size);
    input.read(reinterpret_cast<char*>(header.data()), header_size);
    
    write_uint64(output, header_size);
    output.write(reinterpret_cast<const char*>(header.data()), header_size);

    uint64_t processed_bytes = sizeof(header_size) + header_size;
    uint64_t total_out_size = processed_bytes;

    // 2. Main Loop
    std::vector<Chunk> batch(BATCH_SIZE);
    Timer timer;

    // Initialize buffers to avoid repeated allocation
    for(auto& chunk : batch) {
        chunk.raw_data.resize(CHUNK_SIZE);
        chunk.scratch_buffer.resize(CHUNK_SIZE);
        // Compressed size bound might be larger than input
        chunk.comp_data.resize(ZSTD_compressBound(CHUNK_SIZE));
    }

    bool done = false;
    while (!done) {
        int chunks_in_batch = 0;

        // A. Read Batch (Serial)
        for (int i = 0; i < BATCH_SIZE; ++i) {
            input.read(reinterpret_cast<char*>(batch[i].raw_data.data()), CHUNK_SIZE);
            batch[i].raw_size = input.gcount();
            if (batch[i].raw_size > 0) {
                chunks_in_batch++;
            }
            if (input.eof() || batch[i].raw_size == 0) {
                done = true;
                break;
            }
        }
        if (chunks_in_batch == 0) break;

        // B. Process Batch (Parallel)
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < chunks_in_batch; ++i) {
            Chunk& c = batch[i];
            
            // 1. Shuffle
            // Ensure even size for BF16
            if (c.raw_size % 2 != 0) {
                 // In real app, handle padding. Here we throw.
                 throw std::runtime_error("Chunk size not even (BF16 alignment error)");
            }
            shuffle_bf16(c.raw_data.data(), c.scratch_buffer.data(), c.raw_size);

            // 2. Compress
            // ZSTD_CCtx is NOT thread-safe, so we use a thread_local one or create one here.
            // Creating one per chunk is slightly overhead, but safe. 
            // Ideally, use thread_local ZSTD_CCtx* ctx;
            ZSTD_CCtx* cctx = ZSTD_createCCtx();
            ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, level);
            
            c.comp_size = ZSTD_compressCCtx(cctx, 
                                            c.comp_data.data(), c.comp_data.size(), 
                                            c.scratch_buffer.data(), c.raw_size, 
                                            level);
            
            if (ZSTD_isError(c.comp_size)) {
                // Cannot throw easily inside OMP, handle gracefully or abort
                std::cerr << "ZSTD Error: " << ZSTD_getErrorName(c.comp_size) << std::endl;
                exit(1);
            }
            ZSTD_freeCCtx(cctx);
        }

        // C. Write Batch (Serial)
        for (int i = 0; i < chunks_in_batch; ++i) {
            write_uint64(output, batch[i].raw_size);
            write_uint64(output, batch[i].comp_size);
            output.write(reinterpret_cast<const char*>(batch[i].comp_data.data()), batch[i].comp_size);

            processed_bytes += batch[i].raw_size;
            total_out_size += (16 + batch[i].comp_size);
        }

        print_progress(processed_bytes, total_input_size);
    }

    std::cout << "\nDone in " << timer.elapsed() << "s" << std::endl;
    std::cout << "Ratio: " << std::fixed << std::setprecision(2) 
              << (double)processed_bytes / total_out_size << "x" << std::endl;
}

// --- Decompression Implementation ---
void decompress(const std::string& input_path, const std::string& output_path) {
    std::ifstream input(input_path, std::ios::binary);
    std::ofstream output(output_path, std::ios::binary);
    if (!input || !output) throw std::runtime_error("File I/O error");

    uint64_t total_input_size = get_file_size(input);
    std::cout << "Decompressing with " << omp_get_max_threads() << " threads..." << std::endl;

    // 1. Recover Header
    uint64_t header_size = 0;
    if (!read_uint64(input, header_size)) throw std::runtime_error("Missing header size");
    std::vector<uint8_t> header(header_size);
    input.read(reinterpret_cast<char*>(header.data()), header_size);
    write_uint64(output, header_size);
    output.write(reinterpret_cast<const char*>(header.data()), header_size);

    std::vector<Chunk> batch(BATCH_SIZE);
    // Pre-allocate decent buffers
    for(auto& chunk : batch) {
        chunk.comp_data.resize(ZSTD_compressBound(CHUNK_SIZE)); 
        chunk.raw_data.resize(CHUNK_SIZE);
        chunk.scratch_buffer.resize(CHUNK_SIZE);
    }

    bool done = false;
    Timer timer;

    while (!done) {
        int chunks_in_batch = 0;

        // A. Read Batch Metadata & Data (Serial)
        for (int i = 0; i < BATCH_SIZE; ++i) {
            if (!read_uint64(input, batch[i].raw_size)) {
                done = true; 
                break;
            }
            if (!read_uint64(input, batch[i].comp_size)) throw std::runtime_error("Corrupted chunk header");

            // Ensure buffer capacity
            if (batch[i].comp_data.size() < batch[i].comp_size) 
                batch[i].comp_data.resize(batch[i].comp_size);
            
            // Read compressed data
            input.read(reinterpret_cast<char*>(batch[i].comp_data.data()), batch[i].comp_size);
            if (input.gcount() != static_cast<std::streamsize>(batch[i].comp_size)) 
                throw std::runtime_error("Truncated compressed data");
            
            chunks_in_batch++;
        }
        if (chunks_in_batch == 0) break;

        // B. Process Batch (Parallel)
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < chunks_in_batch; ++i) {
            Chunk& c = batch[i];
            
            if (c.raw_data.size() < c.raw_size) c.raw_data.resize(c.raw_size);
            if (c.scratch_buffer.size() < c.raw_size) c.scratch_buffer.resize(c.raw_size);

            ZSTD_DCtx* dctx = ZSTD_createDCtx();
            size_t d_size = ZSTD_decompressDCtx(dctx, 
                                                c.scratch_buffer.data(), c.raw_size, 
                                                c.comp_data.data(), c.comp_size);
            
            if (ZSTD_isError(d_size)) {
                std::cerr << "ZSTD Decompress Error: " << ZSTD_getErrorName(d_size) << std::endl;
                exit(1);
            }
            ZSTD_freeDCtx(dctx);

            unshuffle_bf16(c.scratch_buffer.data(), c.raw_data.data(), c.raw_size);
        }

        // C. Write Batch (Serial)
        for (int i = 0; i < chunks_in_batch; ++i) {
            output.write(reinterpret_cast<const char*>(batch[i].raw_data.data()), batch[i].raw_size);
        }
        print_progress(input.tellg(), total_input_size);
    }
    
    std::cout << "\nDone in " << timer.elapsed() << "s" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <compress|decompress> <input> <output> [level]" << std::endl;
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