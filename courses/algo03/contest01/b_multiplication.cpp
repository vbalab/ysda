#include <cstring>
#include <cwchar>
#include <fstream>
#include <iostream>
#include <vector>

const std::string kInput = "input.bin";
const std::string kOutput = "output.bin";
constexpr int KBatchRows = 200;
constexpr int KStartIndex = 8;

void MultiplyBlock(const std::vector<char>& line, std::vector<char>& outLine,
                   const std::vector<char>& batch, int startRow, int endRow,
                   int size) {
    for (int row = startRow; row < endRow; ++row) {
        auto elem = line[row];
        int base = size * (row - startRow);
        for (int j = 0; j < size; ++j) {
            outLine[j] += elem * batch[base + j];
        }
    }
}

int ReadSize(std::ifstream& in) {
    std::vector<char> hdr(KStartIndex);
    in.read(hdr.data(), KStartIndex);
    int size = 0;
    std::memcpy(&size, hdr.data(), 4);
    return size;
}

void WriteHeader(std::ofstream& out, int size) {
    std::vector<char> hdr(KStartIndex);
    std::memcpy(hdr.data(), &size, 4);
    std::memcpy(hdr.data() + 4, &size, 4);
    out.write(hdr.data(), KStartIndex);
}

void Process(std::ifstream& in, std::ofstream& out, int size) {
    std::vector<char> line(size);
    std::vector<char> outLine(size);
    std::vector<char> batch(size * KBatchRows);

    for (int i = 0; i < size; ++i) {
        std::streamoff rowPos =
            KStartIndex + static_cast<std::streamoff>(i) * size;
        in.seekg(rowPos, std::ios::beg);
        in.read(line.data(), size);

        for (int start = 0; start < size; start += KBatchRows) {
            int rows = std::min(size - start, KBatchRows);
            std::streamoff batchPos = KStartIndex * 2 +
                                      static_cast<std::streamoff>(size) * size +
                                      static_cast<std::streamoff>(start) * size;
            in.seekg(batchPos, std::ios::beg);
            in.read(batch.data(), static_cast<std::streamsize>(rows) * size);
            MultiplyBlock(line, outLine, batch, start, start + rows, size);
        }

        out.seekp(rowPos, std::ios::beg);
        out.write(outLine.data(), size);
        outLine.assign(size, 0);
    }
}

int main() {
    std::ifstream in(kInput, std::ios_base::binary);
    std::ofstream out(kOutput, std::ios_base::binary);

    int size = ReadSize(in);

    WriteHeader(out, size);
    Process(in, out, size);

    return 0;
}
