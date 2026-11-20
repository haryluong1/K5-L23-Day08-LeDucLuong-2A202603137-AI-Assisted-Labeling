# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202603137 - Lê Đức Lương

> **Chỉ viết phần thay đổi so với v1.0** (không copy cả file). Chốt các câu hỏi mở ở mục 6 của v1.0 và thêm quy tắc mới rút ra từ lỗi của máy / lỗi của bạn. Mỗi ca khó **có ảnh ví dụ** (`img_0xx`) và khớp một dòng trong `qc/calibration_decisions.csv`.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi | Quyết định | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 1 | Xe đạp điện / xe máy điện nhỏ: `bicycle` hay `motorbike`? | Phân loại theo cấu tạo bàn đạp: xe **có bàn đạp ở trục giữa** gán là `bicycle`; xe **không có bàn đạp** (có sàn để chân phẳng và cụm động cơ/bình điện kín như xe tay ga) gán là `motorbike`. | `img_018.jpg`, `img_021.jpg` | `CAL-01` |
| 2 | Khối xe máy đỗ/chạy dày đặc gần/trung cảnh: từng xe hay `ignore_region`? | Nếu xe có chiều cao `>= 20 px` và nhìn thấy rõ đầu xe hoặc bánh xe riêng biệt thì **bắt buộc gán từng xe**; chỉ gom vào `ignore_region` khi cụm `>= 5` xe có chiều cao `< 20 px` ở hậu cảnh hoặc bị che khuất chồng lấn `> 70%` không thể phân định biên từng xe. | `img_005.jpg`, `img_018.jpg`, `img_028.jpg` | `CAL-01` |
| 3 | Hàng nhô ra trên xe đạp / xe ba gác: box và mask xe có gồm hàng không? | Không gộp phần hàng hóa (cây tre, thùng xốp, rổ hàng) nhô ra quá `20%` chiều dài khung cứng của xe vào box hoặc mask của xe; chỉ ôm phần khung xe, thùng xe cố định và bánh xe. | `img_004.jpg`, `img_017.jpg` | `CAL-03` |
| 4 | Người ngồi trong xe ba bánh có mui che: ngưỡng "thấy >= 20%" đo thế nào cho nhất quán? | Thay quy tắc ước lượng `20%` diện tích toàn thân (dễ lệch giữa các người gán) bằng **quy tắc bộ phận giải phẫu**: gán box `person` riêng nếu nhìn thấy rõ **trọn cụm đầu + vai** (hoặc từ ngực trở lên) của người lái/hành khách và box cao `>= 20 px`. | `img_013.jpg`, `img_003.jpg` | `CAL-02` |
| 5 | Kính chắn gió / cửa sổ xe có thuộc mask của xe không? | **Có**: kính chắn gió và khung cửa sổ thuộc mask của `car` / `bus` / `truck` / `three_wheeler`. Tuy nhiên, với xe hở sườn như tuk-tuk (`three_wheeler`), phần cơ thể người lái nhìn thấy trực tiếp qua khoảng hở không có kính phải được loại trừ bằng điểm click âm (`--neg`) để tách riêng với mask `person`. | `img_013.jpg`, `img_017.jpg` | `CAL-02` |
| 6 | Box máy chồng nhau quá nhiều trong ảnh đêm / ảnh đông: gộp hay giữ? | Nếu là **nhiều vật thật đứng chồng lớp** (ví dụ người ngồi trên xe máy, hoặc 2 xe đỗ so le): giữ từng box riêng cho từng cá thể. Nếu là **2 box máy dự đoán trùng lên cùng 1 cá thể** (do đa prompt của YOLOE hoặc NMS sót): xoá box dư, chỉ giữ 1 box khít nhất. | `img_024.jpg`, `img_029.jpg` | `CAL-03` |
| 7 | Có đặt ngưỡng `conf` tối thiểu để tự động chấp nhận box máy không cần nhìn ảnh không? | **Không đặt.** Thực nghiệm ở Part 2–4 cho thấy `YOLO26s` vẫn gán nhầm tuk-tuk thành `car`/`truck` ở `conf > 0.60` (`img_013.jpg`) và `YOLOE-26s` sinh box trùng ở `conf > 0.50`. Thuộc tính `conf` chỉ dùng để ưu tiên thứ tự kiểm tra, mọi box đều phải kiểm bằng mắt. | `img_013.jpg`, `img_002.jpg` | `CAL-03` |

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, box `can_review`)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định của bạn | Quy tắc mới (vào mục nào của v1.0) |
|---|---|---|---|---|
| `img_013.jpg`, `img_003.jpg`, `img_017.jpg` | Một chiếc tuk-tuk / xe ba gác bị sinh 2 box `three_wheeler` chồng nhau hoặc còn sót box `car`/`truck` lệch nhẹ (`IoU < 0.6`) (`CAL-03`, `loi_may`) | `YOLOE-26s` kích hoạt đồng thời theo nhiều prompt (`tuk-tuk`, `cargo tricycle`) hoặc `YOLO26s` bắt phần đầu xe thành `car`/`truck` | Xoá toàn bộ box trùng lặp, chỉ giữ duy nhất 1 box `three_wheeler` ôm sát khung cứng + mái + 3 bánh xe | **Bổ sung Mục 4.1.5 & Mục 5:** Mỗi phương tiện ba bánh chỉ có 1 box `three_wheeler`; khi review phải kiểm tra cặp box chồng nhau quanh mọi xe tuk-tuk/ba gác |
| `img_012.jpg`, `img_020.jpg`, `img_017.jpg` | Người bị cắt ở sát mép trên ảnh (`img_012.jpg` id `#8`), ô tô bị người che khuất một phần ở nền xa (`img_020.jpg` id `#13`) và người ở quá xa cao `< 20 px` (`img_017.jpg` box `#80` cao `10.6 px`) (`CAL-04`, `can_review`) | `YOLO26s` bỏ sót người bị cắt mép trên `img_012.jpg` nhưng lại bắt nhiều box `person` tí hon `< 20 px` ở hậu cảnh `img_017.jpg` | Giữ box ở mép/bị che nếu phần nhìn thấy cao `>= 20 px` và rõ bộ phận (`img_012.jpg`, `img_020.jpg`); xoá hoặc gom vào `ignore_region` mọi box đơn lẻ có chiều cao `< 20 px` (`img_017.jpg` box `#80`) | **Siết Mục 4.1.2, 4.1.3 & 4.1.6:** Box hợp lệ ở mép ảnh hoặc bị che phải có chiều cao hiển thị `>= 20 px`; nghiêm cấm để sót box đơn lẻ `< 20 px` ngoài `ignore_region` |
| `img_029.jpg`, `img_004.jpg`, `img_005.jpg` | Lọt bẫy `wrong_class` (`person` mang nhãn `bicycle` ở `img_029.jpg` `#395`) và 2 bẫy `missing` (`person` ở rìa trái `img_004.jpg`, `motorbike` ở `img_005.jpg`) (`CAL-05`, `bay_lot`) | Pre-label có gài bẫy xoá mất box thật (`missing`) và tráo tên lớp (`wrong_class`) trong cảnh đông đúc khiến người duyệt bị thiên kiến tự động hóa bỏ qua | Bổ sung quy trình bắt buộc: ẩn nhãn máy (`H`) 10 giây đầu để quét 4 vùng biên ảnh tìm vật bị sót, và dùng `Filters` lọc riêng lớp `bicycle`, `three_wheeler`, `truck` trên ảnh đông (`> 15` vật) | **Bổ sung Mục 7 (Quy trình Review chống thiên kiến):** Thực hiện bước ẩn nhãn máy quét rìa ảnh trước và lọc cô lập theo lớp bằng `Filters` trên CVAT |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên | — |
| v1.1 | Chốt 7 câu hỏi mở mục 6 (phân biệt xe đạp điện theo bàn đạp, tiêu chí `ignore_region` cho cụm xe máy `< 20 px`, tiêu chí thấy rõ cụm đầu-vai thay cho `20%` diện tích mơ hồ, quy định kính chắn gió và tách người lái trong mask SAM); bổ sung 3 quy tắc xử lý box `three_wheeler` trùng của YOLOE, siết ngưỡng chiều cao `>= 20 px` và quy trình ẩn nhãn + lọc `Filters` chống thiên kiến tự động hóa. | Rút ra từ thực nghiệm 24 ảnh batch, 4 ảnh gold và 3 bẫy bị lọt (`CAL-01`, `CAL-02`, `CAL-03`, `CAL-04`, `CAL-05`) |
