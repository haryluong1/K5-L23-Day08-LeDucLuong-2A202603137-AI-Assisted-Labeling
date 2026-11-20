# Báo cáo Day 08 — 2A202603137 - Lê Đức Lương

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / bus / truck | `YOLO26s` (`yolo26s.pt`) | `0.30` (mặc định) | Detector đóng 80 lớp COCO bắt nhanh (1.33s/ảnh ở `r05`, `r01`); trên 4 ảnh gold `person` đạt recall tối đa 80% (< 90%, 10 vật) và `bicycle`, `motorbike` (4 vật), `bus`, `truck` có `< 5` vật trong gold nên giữ ngưỡng mặc định `0.30` theo `threshold_sweep.md` |
| car | `YOLO26s` (`yolo26s.pt`) | `0.45` (theo `threshold_sweep`) | Từ `r01`, `r02`, `r05` và `qc/threshold_sweep.md`: ở ngưỡng `0.45`, lớp `car` đạt recall `100%` (`>= 90%`, 6/6 vật gold) và tăng precision lên `55%` (loại bỏ bớt box thừa conf thấp) |
| three_wheeler | `YOLOE-26s-seg` (`yoloe-26s-seg.pt`, prompt: `tuk-tuk`, `cyclo`, `cargo tricycle`) | `0.30` (mặc định) | `YOLO26s` (`r01`, `r02`) không có lớp `three_wheeler` trong COCO nên bỏ sót hoặc gán nhầm thành `car`/`truck`; `YOLOE-26s` (`r04`, `r06`) bắt đúng tuk-tuk/xích lô/ba gác qua prompt chữ (trong 4 ảnh gold có `< 5` vật nên giữ ngưỡng mặc định `0.30`) |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên):
  `python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "car=0.45" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json`
- **Phiên bản SAM bạn chọn** (SAM 1 / 2.1 / 3) và lý do: Chọn **SAM 2.1 Tiny (`sam2.1_t.pt`)** làm model chính (và đối chiếu thêm với **SAM 1 MobileSAM `mobile_sam.pt`** ở Part 5) vì máy chạy CPU (Windows 11, không có GPU rời), `sam2.1_t.pt` (~78 MB) và `mobile_sam.pt` (~40 MB) tốn ít RAM, chạy ổn định song song với Docker CVAT, không cần quyền truy cập gated như SAM 3 nhưng vẫn cho polygon mask bám sát biên vật thể (`r03`: 10/10 mask trên `img_004.jpg` và `img_013.jpg`).
- Lỗi nào **không sửa được bằng đổi ngưỡng** (ví dụ tuk-tuk bị gọi `car`)? Sửa thế nào? Lỗi **lệch hệ quy chiếu nhãn (taxonomy mismatch / closed-set blindness)**: `YOLO26s` chỉ biết 80 lớp COCO nên gọi chiếc tuk-tuk/xe ba gác (trên `img_002.jpg`, `img_011.jpg`, `img_013.jpg`, `img_032.jpg`) là `car` hoặc `truck` với điểm tự tin cao, và khi nối vào SAM (`r03`) thì mask của SAM cũng mang nhãn sai `car` theo box đầu vào. Nâng hay hạ `--conf` chỉ làm mất box hoặc thêm box rác chứ không đổi được tên lớp sang `three_wheeler`. Cách sửa: dùng model open-vocabulary **`YOLOE-26s`** với prompt `"tuk-tuk, cyclo, cargo tricycle" -> three_wheeler` và đặt file `yoloe_all` **đứng trước** `y26_all` trong `merge_prelabels.py` (`--iou 0.6`) để nhãn `three_wheeler` ghi đè box `car`/`truck` trùng vị trí, kết hợp kiểm tra bằng mắt ở bước 3 của checklist review.
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào):
  - **Điểm cao mà sai:** Trên `img_002.jpg` và `img_013.jpg`, `YOLO26s` dự đoán chiếc tuk-tuk là `car` với confidence `0.42–0.75`; trên `img_002.jpg`, `YOLOE-26s` cũng sinh box phụ `three_wheeler` (`conf ~ 0.37`) đè lên xe bán tải ở nền.
  - **Điểm thấp mà đúng:** Trên `img_011.jpg` và `img_016.jpg`, nhiều người đi bộ (`person`) và xe máy (`motorbike`) ở xa hoặc bị che khuất một phần có confidence chỉ `0.25–0.40` (bị mất hoàn toàn khi nâng ngưỡng lên `0.50` ở `r02`, làm mất 32 box `person` và 19 box `motorbike`).
- Giới hạn: chọn ngưỡng trên chính gold (4 ảnh) nên số đo máy trên gold là lạc quan: Tập gold chỉ có **4 ảnh (20 đối tượng tính điểm + 2 `ignore_region`)**, nhiều lớp có dưới 5 mẫu (`motorbike` chỉ 4 vật, `bicycle`/`bus`/`truck`/`three_wheeler` không đủ `>= 5` mẫu để chọn ngưỡng riêng). Hơn nữa, việc vừa dùng 4 ảnh gold để tìm ngưỡng tối ưu trong `threshold_sweep.py` vừa dùng chính 4 ảnh đó để tính Precision/Recall của máy ở Part 4 là đánh giá **in-sample (chọn và chấm trên cùng dữ liệu)**, nên các chỉ số Precision/Recall trên gold (`car` recall 100%) mang tính lạc quan hơn hiệu năng thực tế trên 24 ảnh batch.

## 2. AI error profile (từ `qc/review_stats.md`)

*(Lưu ý: Script `review_stats.py` đã tự động **loại 19 đối tượng nằm quanh 12 vùng bẫy nhân tạo** khỏi bảng profile này vì bẫy do script `inject_traps.py` cấy vào để đo người duyệt, không phản ánh lỗi thật của mô hình AI).*

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | 279 | 279/429 = 65.0% | 429 box máy (sau khi trừ vùng bẫy) |
| edited | 10 | 10/429 = 2.3% | 429 box máy |
| deleted | 129 | 129/429 = 30.1% | 429 box máy |
| added | 7 | 7/296 = 2.4% | 296 đối tượng sau duyệt (sau khi trừ vùng bẫy) |

- Trong số `deleted + added`, có **4 trường hợp là đổi lớp** (máy định vị box đúng chỗ nhưng gán sai tên lớp, ví dụ `car`/`truck` đổi sang `three_wheeler`), nên số vật máy **bỏ sót thật sự** là `added − đổi lớp = 7 − 4 = 3` đối tượng.
- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)?
  1. **`three_wheeler`** (`accepted = 11`, `edited = 4`, `deleted = 9`, `added = 2`): Tỉ lệ phải sửa và xoá rất cao (`13/24 = 54.2%` box máy không dùng nguyên được) vì `YOLOE-26s` dùng 3 prompt chữ (`tuk-tuk`, `cyclo`, `cargo tricycle`) nên hay sinh 2 box chồng lấn lên cùng một chiếc xe ba bánh hoặc khung bao lỏng lẻo ở phần mái che/bánh trước (`img_003.jpg`, `img_004.jpg`, `img_013.jpg`, `img_017.jpg`).
  2. **`person` và `motorbike` ở cảnh đông đúc** (`person`: `deleted = 79/217`; `motorbike`: `deleted = 26/125`): Ở ngưỡng `0.30`, `YOLO26s` bắt nhiều người/xe máy nhỏ `< 20 px` ở nền xa trong 6 ảnh giao thông dày đặc (`img_005.jpg`, `img_018.jpg`, `img_021.jpg`, `img_024.jpg`, `img_028.jpg`, `img_029.jpg`); khi người duyệt gom các cụm xa vào `ignore_region` và xoá các box nhỏ/box nhiễu, số lượng `deleted` tăng cao. Ngoài ra lớp `truck` có `5/5` box bị xoá do máy bắt nhầm xe ba gác/xe tải nhỏ ở nền xa.
- So với gold: precision / recall của **máy** (từ `qc/review_stats.md` mục 2):
  - **Precision = `32.7%`** (`TP / (TP + FP) = 18 / (18 + 37) = 18/55`, KTC 95%: `22–46%`, mẫu số: `55` box máy dự đoán trên 4 ảnh gold). Precision thấp vì trên 4 ảnh gold bạn gán tập trung vào các đối tượng chính rõ ràng (`20` đối tượng tính điểm + `2` vùng `ignore_region`), trong khi máy ở ngưỡng `0.30` vẽ thêm `37` box nhỏ ở nền xa ngoài vùng `ignore_region`.
  - **Recall = `90.0%`** (`TP / (TP + FN) = 18 / (18 + 2) = 18/20`, KTC 95%: `70–97%`, mẫu số: `20` đối tượng gold).

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: **`9/12 = 75.0%`** (KTC 95%: `47–91%`, ngưỡng mục tiêu `80%`). Cụ thể theo từng loại:
  - `loose` (box bị nới lỏng/lệch): **`3/3 (100%)`** — bắt đủ trên `img_008.jpg`, `img_014.jpg`, `img_031.jpg`.
  - `extra` (box ma/thừa): **`3/3 (100%)`** — xoá đủ 3 box giả `truck` trên `img_010.jpg`, `img_022.jpg`, `img_025.jpg`.
  - `wrong_class` (sai nhãn lớp): **`2/3 (66.7%)`** — bắt được trên `img_013.jpg` và `img_034.jpg`, nhưng **lọt 1 bẫy trên `img_029.jpg`** (box `#395` `[652.2, 447.5, 73.7, 157.4]` vốn là `person` bị đổi nhãn thành `bicycle`).
  - `missing` (xoá mất box của vật thật): **`1/3 (33.3%)`** — bắt được chiếc `three_wheeler` lớn bị xoá trên `img_003.jpg`, nhưng **lọt 2 bẫy `missing` trên `img_004.jpg`** (`person` nhỏ `[47.8, 295.7, 40.2, 60.4]` ở sát rìa trái ảnh) **và `img_005.jpg`** (`motorbike` `[177.1, 504.8, 113.0, 249.8]` giữa đám đông xe máy dày đặc).
- **Vì sao lọt 3 bẫy này (bước nào của checklist bị bỏ qua)?**
  - Với 2 bẫy `missing` (`img_004.jpg` và `img_005.jpg`): Đã thực hiện chưa triệt để **Bước 1 của checklist (Ẩn nhãn máy và quét toàn bộ ảnh từ trái sang phải trước khi nhìn box máy)**. Khi mở `img_004.jpg` và `img_005.jpg` (ảnh có mật độ box dày đặc), mắt bị hút ngay vào việc kiểm tra và xoá hàng chục box xanh/đỏ có sẵn của máy ở trung tâm ảnh mà quên quét vùng rìa trái (`img_004.jpg`) hoặc bị ngợp giữa rừng xe máy (`img_005.jpg`). Đây là biểu hiện điển hình nhất của **thiên kiến tự động hóa dạng bỏ sót (omission automation bias)**: con người mặc định tin rằng chỗ nào máy không vẽ box nghĩa là chỗ đó không có vật đáng chú ý.
  - Với 1 bẫy `wrong_class` (`img_029.jpg`): Trên ảnh đông xe (`img_029.jpg`), người đi bộ đứng sát cạnh dòng xe hai bánh nên khi thực hiện **Bước 3 (Kiểm tra nhãn lớp)** đã lướt nhanh theo màu box mà không đối chiếu lại từng nhãn `bicycle` vs `person` trong panel *Objects*.
- **Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt?**
  - Nếu `accepted` rất cao (`> 85–90%`) nhưng bắt bẫy thấp (`< 60%`), điều đó chứng tỏ người duyệt đang rơi vào **automation bias nặng** (chỉ bấm lướt qua ảnh và chấp nhận gợi ý của máy chứ không thật sự kiểm tra).
  - Trong lượt duyệt này, `accepted = 65.0%` (đã chủ động xoá `129` box nhiễu và sửa `10` box) và bắt được **100% (`6/6`) các bẫy hiển thị sẵn trên màn hình (`extra` và `loose`)**, nhưng chỉ bắt được `1/3` bẫy `missing`. Điều này cho thấy mắt người rất nhạy với **lỗi hiện hữu trước mắt** (box thừa nằm vô lý trên đường hoặc box bị lệch viền), nhưng lại rất dễ bị mù trước **lỗi vắng mặt (`missing`)** và **lỗi đổi nhãn trong đám đông (`wrong_class`)** nếu không ẩn nhãn máy để đếm độc lập từ đầu.

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`):
  - **Gán tay (`gold_manual`):** `15 phút / 4 ảnh = 3.75 phút/ảnh` (`22` đối tượng → trung bình `~40.9 giây/đối tượng`).
  - **Duyệt có pre-label (`batch_review`):** `45 phút / 24 ảnh = 1.88 phút/ảnh` (`309` đối tượng → trung bình `~8.7 giây/đối tượng`).
  - **Thiên lệch thời gian (learning effect):** Vì bạn đã gán 4 ảnh gold trước ở Part 1 nên khi sang Part 4 bạn đã thuộc quy tắc taxonomy v1.0 và quen phím tắt CVAT (`N`, `Del`, `Tab`, `Ctrl+S`), do đó tốc độ nhanh gấp đôi (`1.88` vs `3.75 phút/ảnh`, và nhanh gấp `4.7 lần` tính theo mỗi đối tượng) có một phần đóng góp từ sự quen tay chứ không hoàn toàn 100% nhờ AI.
- **Kết luận:** **CÓ đáng dùng, nhưng phải chọn lọc theo lớp và tinh chỉnh lại ngưỡng/quy tắc `ignore_region`**:
  - Với `person`, `motorbike`, `car`: máy giữ nguyên được `279/429 = 65.0%` số box đạt chuẩn `IoU >= 0.9` (`person`: `137` box accepted, `motorbike`: `97` box accepted, `car`: `31` box accepted) và đạt recall `90.0%` trên gold (`18/20`), giúp tiết kiệm hơn một nửa thời gian vẽ tay.
  - Với `three_wheeler`: `YOLOE-26s` giúp phát hiện được vị trí xe ba bánh (`11` box accepted + `4` box edited) mà `YOLO26s` hoàn toàn mù, nhưng vẫn cần người duyệt chỉnh lại viền và xoá box trùng (`9` box deleted).
  - Với `truck`: `0/5` box accepted (`5/5` bị xoá) → không hiệu quả ở ngưỡng `0.30` vì máy hay bắt nhầm xe ba gác/xe tải nhỏ ở xa.
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. **Quy tắc "Ẩn nhãn 10 giây đầu & Quét biên ảnh" (chống sót bẫy `missing`):** Khi mở mỗi ảnh trong CVAT, bắt buộc bấm phím tắt ẩn toàn bộ box (`H` / biểu tượng con mắt ở bảng *Objects*) trong ít nhất **10 giây đầu**, quét mắt theo 4 vùng biên (trái, phải, tiền cảnh, hậu cảnh) và đếm nhẩm số vật lớn trước khi bật nhãn máy lên. *Chỉ số kiểm chứng:* Tỉ lệ bắt bẫy `missing` phải tăng từ `1/3 (33.3%)` lên `>= 2/3` hoặc `3/3`, và mỗi ảnh có mật độ `> 15` vật phải có thời gian dừng tối thiểu `>= 90 giây` trong log.
  2. **Quy tắc "Lọc cô lập theo lớp (`Filters`) cho các lớp dễ nhầm" (chống lọt bẫy `wrong_class`):** Sau khi duyệt xong hình học trên một ảnh đông đúc (`> 20` box như `img_005`, `img_029`), dùng bộ lọc `Filters` của CVAT lọc riêng từng lớp ít gặp hoặc hay bị gán nhầm (`bicycle`, `three_wheeler`, `truck`, `car`) để kiểm tra 100% các box của lớp đó có đúng vật thể thật không. *Chỉ số kiểm chứng:* `100%` box mang nhãn `bicycle`, `three_wheeler`, `truck` được kiểm riêng; tỉ lệ bắt bẫy `wrong_class` đạt `3/3 (100%)` và lỗi `sai_lop` trong mẫu QC giảm về `0`.
  3. **Quy tắc "Khoanh `ignore_region` trước, xoá box nhỏ `< 20 px` sau" (giảm mệt mỏi thị giác trên ảnh dày đặc):** Trên các ảnh giao thông dày đặc, bước đầu tiên là vẽ ngay khung `ignore_region` bao phủ vùng hậu cảnh có xe/người nhỏ `< 20 px`, rồi xoá toàn bộ box máy nằm lọt trong vùng đó trước khi duyệt các vật ở tiền cảnh. *Chỉ số kiểm chứng:* `validate_labels.py --all` đạt **0 cảnh báo** box cao `< 20 px` (loại bỏ cảnh báo kiểu `img_017.jpg box #80 cao 10.6px < 20px`).

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| `img_013.jpg` | `SAM 2.1 Tiny` (`sam2.1_t.pt`) | 3 click (`2 pos`: `400,480`, `600,110`; `1 neg`: `720,300`) | Còn dính một phần đầu/chân người lái và khách ngồi sau (diện tích `275,125 px = 39.4%` ảnh, `1 polygon`) | 1 click âm (`720,300`) mới khoét lõm phần lưng áo người lái chứ chưa tách đứt hoàn toàn người lái khỏi vòm mái tuk-tuk; mask hơi tràn nhẹ xuống bóng đổ bánh trước | `tuktuk_img013_sam2.jpg` |
| `img_013.jpg` | `SAM 1 MobileSAM` (`mobile_sam.pt`) | 3 click (`2 pos`: `400,480`, `600,110`; `1 neg`: `720,300`) | Khoét vùng người lái mạnh hơn (`245,541 px = 35.2%` ảnh) nhưng tách thành `2 polygon` rời nhau | Biên mask ở khung sắt mái vòm và chắn bùn răng cưa hơn `sam2.1_t.pt`, tách mảnh phụ ở phần mui trên | `tuktuk_img013_sam1.jpg` |
| `img_004.jpg` | `SAM 2.1 Tiny` (`sam2.1_t.pt`) | 3 click (`2 pos`: `620,480`, `700,560`; `1 neg`: `465,450`) | Không gồm người xung quanh, nhưng bao gồm cả thùng/khung chở hàng gắn liền xe ba bánh (`86,899 px = 12.4%` ảnh, `1 polygon`) | Điểm âm `465,450` chặn tốt không cho mask lan sang xe/người bên trái, nhưng phần gầm bánh xe tối màu bị hụt nhẹ | `tuktuk_img004_sam2.jpg` |
| `img_017.jpg` | `SAM 2.1 Tiny` (`sam2.1_t.pt`) | 3 click (`2 pos`: `560,420`, `680,500`; `1 neg`: `645,355`) | Khoét tách được phần thân trên người lái nhờ điểm âm `645,355` (`84,062 px = 12.0%` ảnh, `1 polygon`) | Phần chân dưới của người lái nằm sát khung sườn xe có độ tương phản thấp nên vẫn dính một phần nhỏ vào mask thân xe | `tuktuk_img017_sam2.jpg` |

- **Đối chiếu guideline mục 4.2: quy tắc nào giúp, quy tắc nào thiếu?**
  - **Quy tắc giúp ích:** Mục 4.2.1 (độ lệch biên `<= 3 px`, không cần tách chi tiết nan hoa nhỏ hơn bánh xe/tay lái), Mục 4.2.2 (không khoét lỗ thủng `< 5%` diện tích vật) và Mục 4.2.3 (bóng đổ không thuộc vật) giúp xác định rõ vùng mục tiêu khi đặt click dương lên thân và mái xe.
  - **Quy tắc còn thiếu ở v1.0 (đã bổ sung vào v1.1):** Mục 4.2 v1.0 chưa hướng dẫn rõ cách xử lý khi **người lái ngồi lọt bên trong khung mui hở của `three_wheeler`** (`img_013.jpg`, `img_017.jpg`) — nếu chỉ click dương lên đầu xe và mui xe thì SAM luôn gộp cả người lái vào mask xe. Cần quy định rõ phải kết hợp khung giới hạn (`--box`) hoặc đặt tối thiểu `2–3` điểm click âm (`--neg`) dọc theo đầu, lưng và chân người lái để tách rời polygon `person` và `three_wheeler`.
- *(Tuỳ chọn)* **Interactor SAM trong CVAT:** Đã triển khai và chạy thử cả **SAM 2.1 Tiny (`pth-ultralytics-sam2-1-t`)** lẫn **Segment Anything MobileSAM (`pth-facebookresearch-sam-vit-h`)** trực tiếp trong mục **AI Tools → Interactors** của CVAT Docker. Khác biệt so với chạy script offline `sam2_points.py`: trên giao diện CVAT có thể nhìn phản hồi mask tức thì sau từng cú click trái/phải để chêm thêm điểm âm ngay tại chỗ bị tràn, nhưng CVAT tự động chuyển mask thành đa giác nhiều đỉnh nên cần bấm giữ `Shift` hoặc chỉnh lại đỉnh nếu viền bị gãy khúc.

## 6. Calibration (Part 6)

- Số quyết định: **`5` quyết định** trong `submission/qc/calibration_decisions.csv` (`CAL-01` đến `CAL-05`, đầy đủ 4 nhóm nguồn: `cau_hoi_mo`, `loi_may`, `can_review`, `bay_lot`).
- **Quyết định quan trọng nhất:** **`CAL-03` (`nguon = loi_may` trên `img_013.jpg`, `img_003.jpg`, `img_017.jpg`)** và **`CAL-05` (`nguon = bay_lot` trên `img_029.jpg`, `img_004.jpg`, `img_005.jpg`)**:
  - `CAL-03`: Xử lý lỗi đặc trưng của mô hình open-vocabulary `YOLOE-26s` khi dùng nhiều prompt (`tuk-tuk`, `cyclo`, `cargo tricycle`) sinh ra nhiều box `three_wheeler` chồng đè lên cùng một chiếc xe hoặc gộp cả hàng hóa nhô quá `20%`.
  - `CAL-05`: Đưa quy trình ẩn nhãn máy 10 giây đầu và lọc cô lập theo lớp (`Filters` cho `bicycle`, `three_wheeler`, `truck`) vào quy chuẩn bắt buộc để chặn lỗi bỏ sót vật ở rìa ảnh và lỗi sai nhãn giữa đám đông.
- **Quy tắc mới đã vào `guideline_v1.1_calibrated.md`:**
  1. Chốt toàn bộ 7 câu hỏi mở của mục 6 v1.0: phân biệt `bicycle` vs `motorbike` (xe điện) theo cấu tạo bàn đạp (`CAL-01`); chỉ gom `ignore_region` khi cụm `>= 5` xe có chiều cao `< 20 px` hoặc bị che khuất chồng lấn `> 70%` (`CAL-01`); thay ngưỡng diện tích `20%` cảm tính bằng tiêu chí nhìn thấy trọn cụm **đầu + vai** (`>= 20 px`) cho người ngồi trong xe có mui (`CAL-02`); kính chắn gió thuộc mask xe nhưng phải đặt click âm tách cơ thể người lái (`CAL-02`).
  2. Mỗi xe ba bánh chỉ giữ duy nhất 1 box `three_wheeler` ôm khung cứng, xoá mọi box trùng lặp do đa prompt của YOLOE (`CAL-03`).
  3. Vật bị cắt ở mép ảnh hoặc bị che vẫn gán nếu phần nhìn thấy cao `>= 20 px` (`img_012.jpg`, `img_020.jpg`), nhưng tuyệt đối xoá hoặc đưa vào `ignore_region` các box đơn lẻ cao `< 20 px` (`img_017.jpg` box `#80` cao `10.6 px`) (`CAL-04`).
- **Nguyên nhân gốc:** Kết hợp cả 3 yếu tố:
  1. **Guideline v1.0 cố ý để mở** (`CAL-01`, `CAL-02`: chưa định nghĩa xe điện, ranh giới `ignore_region` ở trung cảnh và cách đo `20%` diện tích bị che).
  2. **Đặc tính lỗi hệ thống của mô hình AI** (`CAL-03`: `YOLO26s` mù lớp `three_wheeler`, `YOLOE-26s` sinh box trùng từ đa prompt, `YOLO26s` bắt nhiều box `< 20 px` ở hậu cảnh).
  3. **Thiên kiến tự động hóa của người duyệt** (`CAL-05`: bị cuốn vào sửa các box máy vẽ sẵn ở giữa ảnh mà quên quét biên ảnh và không lọc kiểm tra riêng từng lớp trong cảnh đông đúc).

## 7. QC và quyết định (Part 6)

- Mẫu: **`5` ảnh** (`0` ảnh `can_review` trong batch + `5` ảnh ngẫu nhiên với `seed = 42`: `img_003.jpg`, `img_006.jpg`, `img_014.jpg`, `img_031.jpg`, `img_035.jpg`), tổng cộng **`55` đối tượng** (trên tổng lô `24` ảnh batch, `309` đối tượng); người kiểm: `LeDucLuong` (**hạn chế**: người kiểm QC trùng với người duyệt ở Part 4 nên dễ lặp lại cùng điểm mù chủ quan).
- Lỗi theo **đối tượng**: **`4 / 55 = 7.3%`** (mẫu số: `55` đối tượng trong 5 ảnh mẫu QC) · Lỗi theo **ảnh**: **`4 / 5 = 80.0%`** (mẫu số: `5` ảnh mẫu QC; ngưỡng Release quy định trong `taxonomy.json` là `error_rate_img_max <= 2%`).
- **Top lỗi** (từ `submission/qc/qc_review.csv` và `qc_report_auto.md`) + ảnh ví dụ:
  1. `sai_vi_tri_kich_thuoc` (**`2` lỗi**): Trên **`img_006.jpg`**, box `#31 motorbike` bao gộp cả cụm 3 chiếc xe máy đỗ sát nhau thay vì tách từng xe; trên **`img_031.jpg`**, box `#268 person` chỉ bao phần thân trên người bán hàng rong tới rổ trái cây, thiếu phần chân nhìn thấy rõ bên dưới khung xe đạp `#266`.
  2. `thua` (**`1` lỗi**): Trên **`img_003.jpg`**, còn sót box nhỏ `#6 person` chồng đè lên phần nón/đầu của người lái xe ba gác `#9 person` (2 box `person` trùng trên cùng 1 người).
  3. `bo_sot` (**`1` lỗi**): Trên **`img_035.jpg`**, bỏ sót 1 xe máy + người lái đang đi vào ở sát mép phải ảnh (ngoài ra trên `img_006.jpg` cũng có 1 người mặc áo xám đứng quay lưng cạnh `#34` bị sót).
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): Với `4/5` ảnh có lỗi (`80.0%`), cận trên khoảng tin cậy 95% (Wilson) của tỉ lệ ảnh lỗi thực tế lên tới **`96%`**. Quan trọng hơn, với cỡ mẫu chỉ `n = 5` ảnh, ngay cả khi quan sát thấy `0/5` lỗi (`0%`) thì cận trên Wilson 95% vẫn là `~43%` (lớn hơn rất nhiều so với ngưỡng `2%`). Do đó, mẫu QC nhỏ `5` ảnh chỉ có tác dụng **phát hiện nhanh lỗi lớn để chặn lô lỗi (reject/rework)** chứ về mặt thống kê **không thể chứng minh** lô đạt chuẩn `< 2%`.
- **Script kiểm được / không kiểm được:**
  - *Kiểm tự động được:* Cấu trúc JSON COCO, tên lớp hợp lệ theo `taxonomy.json`, số lượng ảnh, chiều cao box `< 20 px` (`validate_labels.py` đã phát hiện đúng box `#80 person` cao `10.6 px < 20 px` trên `img_017.jpg`), tỉ lệ bắt 12 bẫy đã biết (`review_stats.py`).
  - *Script KHÔNG tự kiểm được:* Box có ôm sát đủ 4 cạnh vật thể hay bị cắt cụt chân như `#268 person` trên `img_031.jpg`; vật thật bị cả máy lẫn người cùng bỏ sót (như người áo xám trên `img_006.jpg` hay xe ở mép phải `img_035.jpg`); box gộp nhiều xe làm một (`#31 motorbike` trên `img_006.jpg`); và người duyệt có thực sự ẩn nhãn quét ảnh theo checklist hay không.

**Quyết định: `Rework`**, ngày `2026-11-20`.
- **Lý do (đi theo đúng cây quyết định `Hold → Rework → Release`):**
  1. *Kiểm mức `Hold`:* Bộ dữ liệu 24 ảnh không bị hỏng và các điểm mơ hồ của guideline v1.0 đã được chốt đầy đủ thành quy tắc rõ ràng trong `guideline_v1.1_calibrated.md` (`guideline_gap = 0`, `data_problem = 0`) → **Không Hold**.
  2. *Kiểm mức `Rework`:* Tỉ lệ ảnh có lỗi trong mẫu QC là **`4/5 = 80.0% > 2%`** (lỗi theo đối tượng `4/55 = 7.3%`), đồng thời tỉ lệ bắt bẫy ở Part 4 là **`9/12 = 75.0% < 80%`** (lọt 2 bẫy `missing` trên `img_004.jpg`, `img_005.jpg` và 1 bẫy `wrong_class` trên `img_029.jpg`), cộng với 1 cảnh báo kích thước `< 20 px` trên `img_017.jpg` (box `#80`). Các chỉ số này vi phạm ngưỡng Release nên bắt buộc chọn **`Rework`**.
- **Nếu Rework / Hold, việc cần làm tiếp:**
  1. Mở lại Task `batch-review` trên CVAT, sửa ngay 4 ảnh lỗi phát hiện qua QC (`img_003.jpg` xoá box trùng `#6`; `img_006.jpg` tách box `#31` và vẽ bù người áo xám; `img_031.jpg` kéo đáy box `#268` xuống hết chân; `img_035.jpg` vẽ bù xe máy ở mép phải).
  2. Sửa 3 ảnh bị lọt bẫy (`img_004.jpg` thêm `person` ở rìa trái; `img_005.jpg` thêm `motorbike` bị thiếu; `img_029.jpg` đổi box `#395` từ `bicycle` về `person`) và xoá box `#80` cao `10.6 px < 20 px` trên `img_017.jpg`.
  3. Áp dụng quy tắc lọc `Filters` theo lớp và ẩn nhãn 10 giây đầu trên toàn bộ các ảnh đông xe còn lại, xuất thành `submission/labels/qc_fixed.json`, chạy lại `build_release.py` và rút mẫu QC vòng 2 với seed mới.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào; "không dùng" nếu không): Dùng trợ lý AI (Antigravity) để hỗ trợ kiểm tra lỗi cài đặt môi trường Python 3.14 (`requirements-local.txt`, `ultralytics`, xử lý tương thích `pkg_resources` cho `openai-clip`), cấu hình container Nuclio cho CVAT Docker và đối chiếu cấu trúc báo cáo. Toàn bộ việc gán nhãn 4 ảnh gold, duyệt 24 ảnh batch trên CVAT và kiểm tra từng ảnh mẫu QC (`work/qc/review/*.jpg`) đều được thực hiện và kiểm chứng trực tiếp bằng mắt trên ảnh và đầu ra của các script `validate_labels.py`, `review_stats.py`, `qc_report.py`.
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? Không — 4 ảnh gold (`img_001.jpg`, `img_012.jpg`, `img_020.jpg`, `img_036.jpg`) được gán tay hoàn toàn độc lập trên CVAT (Task `gold-manual`, 22 đối tượng trong 15 phút, trung bình 3.75 phút/ảnh) trước khi chạy bất kỳ model pre-label nào. Các ca khó / phân vân quy tắc đã gắn cờ `can_review` gồm: `img_012.jpg` (người `#8` ở rìa trên bị cắt một phần cơ thể), `img_020.jpg` (xe `#13` ở nền xa bị che khuất một phần phía sau người đi bộ) và `img_036.jpg` (cụm xe/người ở xa biên ~20 px gom vào `ignore_region`).
- Máy / GPU / CVAT dùng; vướng mắc cài đặt: Máy Windows 11 chạy CPU (không có GPU rời), Python 3.14.10, CVAT Docker `v2.74.1` kèm Nuclio serverless (`YOLO v26s`, `YOLOE v26s-seg`, `SAM 2.1 Tiny`, `MobileSAM`). Vướng mắc đã xử lý: cài bổ sung `numpy`, `opencv-python-headless`, `matplotlib`, `pyyaml`, `torch`, `ultralytics`, `openai-clip` trên Python 3.14 và chuẩn hóa ký tự xuống dòng `\n` trong Nuclio local store.
- Dùng cache `data/cache/` thay vì tự chạy? Không — đã tự chạy trực tiếp toàn bộ 6 lượt `prelabel.py` (`r01` đến `r06` trong `work/runs.csv`) và 4 lượt `sam2_points.py` trên CPU của máy.
- Lỗi / bug script đã gặp (nếu có): Trên Python 3.14 với `setuptools >= 81`, gói `pkg_resources` bị gỡ bỏ khiến lệnh `from pkg_resources import packaging` trong thư viện `openai-clip` (khi chạy `YOLOE`) báo lỗi `ModuleNotFoundError: No module named 'pkg_resources'`. Đã sửa trong `scripts/prelabel.py` (dòng 121–126) bằng cách ánh xạ `sys.modules["pkg_resources"].packaging = packaging` trước khi nạp `YOLOE`.
- Một điều bạn làm khác lần sau: Khi duyệt các ảnh đường phố đông đúc trên CVAT, sẽ luôn bấm phím `H` ẩn toàn bộ nhãn máy trong 10 giây đầu để quét bốn cạnh rìa ảnh và khoanh `ignore_region` trước, sau đó dùng `Filters` lọc riêng từng lớp (`bicycle`, `three_wheeler`) — cách này sẽ giúp bắt trọn các bẫy `missing` ở rìa ảnh (`img_004.jpg`) và `wrong_class` trong đám đông (`img_029.jpg`).
