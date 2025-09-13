#include <bits/stdc++.h>

struct Rect {
    int llx;
    int lly;
    int urx;
    int ury;
    int color;
};

void Input(int& width_of_sheet, int& height_of_sheet, int& number_of_rectangles,
           std::vector<Rect>& rects) {
    std::cin >> width_of_sheet >> height_of_sheet >> number_of_rectangles;
    rects.resize(number_of_rectangles);
    for (int i = 0; i < number_of_rectangles; i++) {
        std::cin >> rects[i].llx >> rects[i].lly >> rects[i].urx >>
            rects[i].ury >> rects[i].color;
    }
}

void CompressCoordinates(int width_of_sheet, int height_of_sheet,
                         const std::vector<Rect>& rects, std::vector<int>& xs,
                         std::vector<int>& ys) {
    xs.push_back(0);
    xs.push_back(width_of_sheet);
    ys.push_back(0);
    ys.push_back(height_of_sheet);

    for (const auto& rect : rects) {
        xs.push_back(rect.llx);
        xs.push_back(rect.urx);
        ys.push_back(rect.lly);
        ys.push_back(rect.ury);
    }

    std::sort(xs.begin(), xs.end());
    xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
    std::sort(ys.begin(), ys.end());
    ys.erase(std::unique(ys.begin(), ys.end()), ys.end());
}

void FillGrid(const std::vector<Rect>& rects, const std::vector<int>& xs,
              const std::vector<int>& ys, std::vector<std::vector<int>>& grid) {
    auto cx = [&](int vx) {
        int idx = static_cast<int>(std::lower_bound(xs.begin(), xs.end(), vx) -
                                   xs.begin());
        return idx;
    };
    auto cy = [&](int vy) {
        int idx = static_cast<int>(std::lower_bound(ys.begin(), ys.end(), vy) -
                                   ys.begin());
        return idx;
    };

    for (const auto& rect : rects) {
        int x1 = cx(rect.llx);
        int x2 = cx(rect.urx);
        int y1 = cy(rect.lly);
        int y2 = cy(rect.ury);

        for (int yy = y1; yy < y2; yy++) {
            for (int xx = x1; xx < x2; xx++) {
                grid[yy][xx] = rect.color;
            }
        }
    }
}

void ComputeArea(const std::vector<int>& xs, const std::vector<int>& ys,
                 const std::vector<std::vector<int>>& grid,
                 std::map<int, long long>& area_map) {
    int y_size = static_cast<int>(ys.size());
    int x_size = static_cast<int>(xs.size());

    for (int yy = 0; yy < y_size - 1; yy++) {
        long long dy = ys[yy + 1] - ys[yy];
        for (int xx = 0; xx < x_size - 1; xx++) {
            int color_index = grid[yy][xx];
            long long dx = xs[xx + 1] - xs[xx];
            long long cell_area = dx * dy;
            area_map[color_index] += cell_area;
        }
    }

    for (auto it = area_map.begin(); it != area_map.end();) {
        if (it->second == 0) {
            it = area_map.erase(it);
        } else {
            ++it;
        }
    }
}

void Output(const std::map<int, long long>& area_map) {
    for (const auto& kv : area_map) {
        std::cout << kv.first << " " << kv.second << "\n";
    }
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int width_of_sheet;
    int height_of_sheet;
    int number_of_rectangles;
    std::vector<Rect> rects;
    Input(width_of_sheet, height_of_sheet, number_of_rectangles, rects);

    std::vector<int> xs;
    std::vector<int> ys;
    CompressCoordinates(width_of_sheet, height_of_sheet, rects, xs, ys);

    int x_size = static_cast<int>(xs.size());
    int y_size = static_cast<int>(ys.size());

    std::vector<std::vector<int>> grid(y_size - 1,
                                       std::vector<int>(x_size - 1, 1));
    FillGrid(rects, xs, ys, grid);

    std::map<int, long long> area_map;
    ComputeArea(xs, ys, grid, area_map);

    Output(area_map);

    return 0;
}
