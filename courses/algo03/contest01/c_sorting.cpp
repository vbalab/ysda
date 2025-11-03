#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

const std::string kInput = "input.bin";
const std::string kOutput = "output.bin";
constexpr size_t kArrayOffset = 8;
constexpr size_t kValueSize = 8;

// use separate sizes for initial sort and merging
constexpr size_t kSortChunkElems = 1'000;     // ~720 KB (fits < 800 KB)
constexpr size_t kMergeBufElems = 8'192 / 16;  // 3 * 8'192 * 8 = 192 KB

int64_t ReadNumberOfElements(std::ifstream& fin) {
    int64_t size;
    char header[kArrayOffset];
    fin.seekg(0, std::ios::beg);
    fin.read(header, kArrayOffset);
    std::memcpy(&size, header + 0, kArrayOffset);
    return size;
}

void WriteHeader(std::ofstream& fout, int64_t n) {
    char header[kArrayOffset];
    std::memcpy(header + 0, &n, kArrayOffset);
    fout.seekp(0, std::ios::beg);
    fout.write(header, kArrayOffset);
}

struct Run {
    std::string path;
    std::size_t count;
};

void SortBlockInMemory(std::ifstream& fin, const std::string& path,
                       std::size_t offset_bytes, std::size_t n_elements) {
    std::vector<int64_t> buffer(n_elements);
    fin.seekg(static_cast<std::streamoff>(offset_bytes), std::ios::beg);
    fin.read(reinterpret_cast<char*>(buffer.data()),
             static_cast<std::streamsize>(n_elements * kValueSize));
    std::sort(buffer.begin(), buffer.end());
    std::ofstream fout(path, std::ios::binary | std::ios::trunc);
    fout.seekp(0, std::ios::beg);
    fout.write(reinterpret_cast<const char*>(buffer.data()),
               static_cast<std::streamsize>(n_elements * kValueSize));
}

Run MergeTwoRuns(const Run& a, const Run& b, const std::string& out_path) {
    std::ifstream fa(a.path, std::ios::binary);
    std::ifstream fb(b.path, std::ios::binary);
    std::ofstream fo(out_path, std::ios::binary | std::ios::trunc);

    std::vector<int64_t> ba(kMergeBufElems);
    std::vector<int64_t> bb(kMergeBufElems);
    std::vector<int64_t> bo(kMergeBufElems);

    std::size_t ia = 0, na = 0, ib = 0, nb = 0, io = 0;
    std::size_t left_a = a.count, left_b = b.count;

    auto refill_a = [&] {
        if (!left_a) {
            na = 0;
            return;
        }
        std::size_t take = std::min(kMergeBufElems, left_a);
        fa.read(reinterpret_cast<char*>(ba.data()),
                static_cast<std::streamsize>(take * kValueSize));
        na = take;
        ia = 0;
        left_a -= take;
    };
    auto refill_b = [&] {
        if (!left_b) {
            nb = 0;
            return;
        }
        std::size_t take = std::min(kMergeBufElems, left_b);
        fb.read(reinterpret_cast<char*>(bb.data()),
                static_cast<std::streamsize>(take * kValueSize));
        nb = take;
        ib = 0;
        left_b -= take;
    };
    auto flush_o = [&] {
        if (io) {
            fo.write(reinterpret_cast<const char*>(bo.data()),
                     static_cast<std::streamsize>(io * kValueSize));
            io = 0;
        }
    };

    refill_a();
    refill_b();

    while (na || nb || left_a || left_b) {
        if (ia == na && left_a) refill_a();
        if (ib == nb && left_b) refill_b();
        if (ia == na && ib == nb) break;

        int64_t v;
        if (ia < na && ib < nb)
            v = (ba[ia] <= bb[ib]) ? ba[ia++] : bb[ib++];
        else if (ia < na)
            v = ba[ia++];
        else
            v = bb[ib++];

        bo[io++] = v;
        if (io == kMergeBufElems) flush_o();
    }
    flush_o();
    return Run{out_path, a.count + b.count};
}

void CopyRunToOutput(const Run& run, std::ofstream& fout) {
    std::ifstream fr(run.path, std::ios::binary);
    std::vector<char> buf(1 * 1024);  // 128 KB
    fout.seekp(static_cast<std::streamoff>(kArrayOffset), std::ios::beg);
    while (fr) {
        fr.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        std::streamsize got = fr.gcount();
        if (got > 0) fout.write(buf.data(), got);
    }
}

void SortArray(std::ifstream& fin, std::ofstream& fout, int64_t n_elements) {
    std::vector<Run> runs;
    std::size_t total_blocks = static_cast<std::size_t>(
        (n_elements + kSortChunkElems - 1) / kSortChunkElems);

    for (std::size_t i = 0; i < total_blocks; ++i) {
        std::size_t block_elems = static_cast<std::size_t>(std::min<int64_t>(
            kSortChunkElems,
            n_elements - static_cast<int64_t>(i) *
                             static_cast<int64_t>(kSortChunkElems)));
        std::string path = "run_" + std::to_string(i) + ".bin";
        std::size_t offset = kArrayOffset + i * kSortChunkElems * kValueSize;
        SortBlockInMemory(fin, path, offset, block_elems);
        runs.push_back(Run{path, block_elems});
    }

    std::size_t pass = 0;
    while (runs.size() > 1) {
        std::vector<Run> next;
        next.reserve((runs.size() + 1) / 2);
        for (std::size_t i = 0; i + 1 < runs.size(); i += 2) {
            std::string path = "merge_" + std::to_string(pass) + "_" +
                               std::to_string(i / 2) + ".bin";
            Run merged = MergeTwoRuns(runs[i], runs[i + 1], path);
            next.push_back(merged);
        }
        if (runs.size() % 2 == 1) next.push_back(runs.back());
        runs.swap(next);
        ++pass;
    }

    if (!runs.empty()) CopyRunToOutput(runs.front(), fout);
}

int main() {
    std::ifstream fin(kInput, std::ios::binary);
    std::ofstream fout(kOutput, std::ios::binary | std::ios::trunc);

    int64_t n_elements = ReadNumberOfElements(fin);
    WriteHeader(fout, n_elements);
    SortArray(fin, fout, n_elements);

    fout.flush();
    return 0;
}
