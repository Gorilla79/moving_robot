# moving_robot

SLAM 지도와 실제 거리를 기반으로 path_planning의 거리 값 구하기
</br>

yaml파일의 resolution 활용</br>
이전 grid_map 과정을 끝내고 해당 과정에서 추출한 result_grid_test_size_min.csv를 기반으로 내용을 수행</br>
</br>
(지도가 세로로 평평하게 출력되었을때 다시 측정해야 됨)</br>
[실제 Waypoint에 따른 거리]</br>
Path 1의 거리(Way Point 1 -> Way Point2) : 39.5m </br>
출력된 Path1의 거리 : 39.20m </br>
</br>
Path 2의 거리(Way Point 2 -> Way Point3) : ?m </br>
출력된 Path2의 거리 : 5.10m </br>
</br>
Path 3의 거리(Way Point 3 -> Way Point4) : ?m </br>
출력된 Path3의 거리 : 52.70m </br>
</br>
Path 4의 거리(Way Point 4 -> Way Point5) : ?m </br>
출력된 Path4의 거리 : 9.40m </br>
</br>
Path 5의 거리(Way Point 5 -> Way Point6) : ?m </br>
출력된 Path5의 거리 : 14.90m </br>
</br>
Path 6의 거리(Way Point 6 -> Way Point7) : ?m </br>
출력된 Path6의 거리 : 20.10m </br>
</br>
Path 7의 거리(Way Point 7 -> Way Point8) : ?m </br>
출력된 Path7의 거리 : 61.80m </br> 
</br>
Path 8의 거리(Way Point 8 -> Way Point9) : ?m </br>
출력된 Path8의 거리 : 6.20m </br>
</br>
![image](https://github.com/user-attachments/assets/0c8563fa-b79a-4478-b249-47fc41a75163)

---

# 내용 정리

# 그리드 맵 경로 계획 및 거리 계산

이 저장소는 A* (A-star) 가중 알고리즘을 사용하여 그리드 기반 경로 계획을 수행하는 Python 코드를 포함합니다. 주요 기능은 다음과 같습니다:

1. **벽으로부터 거리 계산**:
   - `distance_transform_edt` 함수를 사용하여 각 셀의 벽으로부터의 유클리드 거리를 계산합니다.

2. **최단 경로 계획**:
   - 가중치를 활용한 A* 알고리즘을 사용하여 선택된 웨이포인트 간의 최단 경로를 찾습니다.

3. **웨이포인트 및 경로 시각화**:
   - 그리드 맵, 경로, 선택된 웨이포인트를 시각화합니다.

4. **전역 좌표 변환**:
   - 맵 메타데이터를 사용하여 픽셀 기반 좌표를 전역 좌표로 변환합니다.

---

## 사전 준비

다음 Python 라이브러리가 설치되어 있어야 합니다:
- `numpy`
- `pandas`
- `matplotlib`
- `scipy`

---

## 기능

### 1. 벽으로부터 거리 계산

`scipy.ndimage`의 `distance_transform_edt` 함수를 사용하여 각 그리드 맵 셀에서 벽까지의 유클리드 거리를 계산합니다.

#### 수식:

각 셀 $(x, y)$에 대해:

$$
D(x, y) = \sqrt{(x_\text{wall} - x)^2 + (y_\text{wall} - y)^2}
$$

여기서 $(x_\text{wall}, y_\text{wall})$ 은 가장 가까운 벽을 나타냅니다.

---

### 2. A* 가중 알고리즘

가중치가 적용된 A* 알고리즘을 구현하여 웨이포인트 간 최단 경로를 찾습니다.

#### 단계:
1. 휴리스틱 함수 정의 (유클리드 거리):

$$h(x, y) = \sqrt{(x_\text{goal} - x)^2 + (y_\text{goal} - y)^2}$$

2. 이웃 셀의 비용 업데이트:

$$f(x, y) = g(x, y) + h(x, y)$$
   
   여기서:
   - $g(x, y)$ : 셀 $(x, y)$ 까지의 실제 비용
   - $h(x, y)$ : 셀 $(x, y)$ 에서 목표까지의 휴리스틱 추정값

#### 코드:
```python
from heapq import heappush, heappop

def heuristic(a, b):
    return distance.euclidean(a, b)

def astar_weighted(grid, weights, start, goal):
    rows, cols = grid.shape
    open_set = []
    heappush(open_set, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    cost_so_far = {start: 0}

    while open_set:
        _, current_cost, current = heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # 경로를 역순으로 반환

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor] == 1:
                new_cost = current_cost + weights[neighbor]
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal)
                    heappush(open_set, (priority, new_cost, neighbor))
                    came_from[neighbor] = current

    return []  # 경로를 찾을 수 없는 경우
```

---

### 3. 경로 거리 계산

주어진 경로의 총 거리를 계산합니다.

#### 수식:
경로가 $(x_i, y_i)$ 로 구성되어 있을 때:

$$
\text{거리} = \sum_{i=1}^{n-1} \sqrt{(x_{i+1} - x_i)^2 + (y_{i+1} - y_i)^2} \times \text{resolution}
$$

#### 코드:
```python
def calculate_path_distance(path, resolution):
    total_distance = 0.0
    for i in range(len(path) - 1):
        current = path[i]
        next_point = path[i + 1]
        pixel_distance = distance.euclidean(current, next_point)
        step_distance = resolution * pixel_distance
        total_distance += step_distance
    return total_distance
```

---

### 4. 전역 좌표 변환

맵 메타데이터(해상도 및 원점)를 사용하여 픽셀 기반 좌표를 전역 좌표로 변환합니다.

#### 수식:

$$
\text{global_x} = \text{pixel_x} \times \text{resolution} + \text{origin}[0]
$$

$$
\text{global_y} = \text{pixel_y} \times \text{resolution} + \text{origin}[1]
$$

#### 코드:
```python
def calculate_global_coordinates(path, resolution, origin):
    global_coordinates = []
    for point in path:
        global_x = point[1] * resolution + origin[0]
        global_y = point[0] * resolution + origin[1]
        global_coordinates.append((global_x, global_y))
    return global_coordinates
```

---

### 5. 시각화

`matplotlib`을 사용하여 그리드 맵, 경로 및 웨이포인트를 시각화합니다.

#### 출력 예시:
- 경로는 색상으로 구분됩니다.
- 웨이포인트는 번호가 매겨집니다.

#### 코드:
```python
fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(grid_array, cmap='gray', origin='upper')

# 경로 시각화
for idx, path in enumerate(adjusted_paths):
    path_x, path_y = zip(*path)
    ax.plot(path_y, path_x, label=f"Path {idx + 1}")

# 웨이포인트 시각화
for wp, coord in enumerate(clicked_waypoints, 1):
    ax.scatter(coord[1], coord[0], color='red')
    ax.text(coord[1], coord[0], str(wp), color='white', ha='center', va='center')

ax.set_title("그리드 맵과 경로 및 웨이포인트")
ax.legend()
plt.show()
```

---

## 실행 방법
1. 그리드 맵 CSV 파일 및 관련 YAML 파일을 적절한 디렉토리에 배치합니다.
2. 스크립트의 `grid_file_path` 변수를 업데이트합니다.
3. 스크립트를 실행하고 맵에서 웨이포인트를 선택합니다.

---

## 예시 YAML 메타데이터
```yaml
image: result.pgm
resolution: 0.1
origin: [-2.94, -4.9, 0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.25
