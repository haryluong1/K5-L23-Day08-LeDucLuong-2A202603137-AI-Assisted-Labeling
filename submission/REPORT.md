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

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | <<>> | <<>> | <<box máy>> |
| edited | <<>> | <<>> | <<>> |
| deleted | <<>> | <<>> | <<>> |
| added | <<>> | <<>> | <<đối tượng sau duyệt>> |

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)? <<>>
- So với gold: precision / recall của **máy** <<>> (mẫu số: <<đối tượng gold>>)

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: <<.../12>>; loại nào lọt: <<>>; vì sao (bạn đã bỏ bước nào của checklist)? <<>>
- Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt? <<>>

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay <<... phút/ảnh>> vs duyệt <<... phút/ảnh>>. Thiên lệch (bạn gán gold trước nên đã quen cảnh): <<>>
- Kết luận: <<có / không / chỉ cho lớp nào>>, dựa trên <<số liệu>>
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. <<>>
  2. <<>>
  3. <<>>

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| <<img_0xx>> | <<>> | <<>> | <<>> | <<>> | <<>> |

- Đối chiếu guideline mục 4.2: quy tắc nào giúp, quy tắc nào thiếu? <<>>
- *(Tuỳ chọn)* Interactor SAM trong CVAT: làm được / không (lỗi gì, đã thử gì); khác gì offline: <<>>

## 6. Calibration (Part 6)

- Số quyết định: <<>> (`calibration_decisions.csv`); quan trọng nhất (dẫn `CAL-xx`): <<>>
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: <<>>
- Nguyên nhân gốc: <<guideline mơ hồ / chưa nắm quy tắc / máy>>

## 7. QC và quyết định (Part 6)

- Mẫu: <<n>> ảnh (<<m>> ảnh `can_review` + <<k>> ngẫu nhiên, seed 42); người kiểm: bạn (**hạn chế**: bạn vừa là người duyệt).
- Lỗi theo **đối tượng**: <<.../... = ...%>> · theo **ảnh**: <<.../... = ...%>> (ngưỡng Release ≤ 2%)
- Top lỗi (trong: bỏ sót, thừa, sai lớp, sai vị trí/kích thước, sai thuộc tính, sai mask) + một ảnh ví dụ mỗi loại: <<>>
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): <<>>
- Script kiểm được / không kiểm được (ví dụ ôm sát, cùng bỏ sót với máy, theo checklist thật): <<>>

**Quyết định: <<Release / Hold / Rework>>**, ngày <<YYYY-MM-DD>>. Lý do (đi theo Hold → Rework → Release): <<>>
Nếu Rework / Hold, việc cần làm tiếp: <<>>

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào; "không dùng" nếu không): <<>>
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? Không — 4 ảnh gold (`img_001.jpg`, `img_012.jpg`, `img_020.jpg`, `img_036.jpg`) được gán tay hoàn toàn độc lập trên CVAT (Task `gold-manual`, 22 đối tượng trong 15 phút, trung bình 3.75 phút/ảnh) trước khi chạy bất kỳ model pre-label nào. Các ca khó / phân vân quy tắc đã gắn cờ `can_review` gồm: `img_012.jpg` (phương tiện ba bánh/tuk-tuk góc nhìn nghiêng dễ nhầm với `car` và người ở rìa trên bị cắt một phần cơ thể), `img_020.jpg` (xe ở nền xa bị che khuất một phần phía sau người đi bộ) và `img_036.jpg` (cụm xe/người ở xa biên ~20 px gom vào `ignore_region`).
- Máy / GPU / CVAT dùng; vướng mắc cài đặt: Máy Windows 11 chạy CPU (không có GPU rời), CVAT Docker `v2.74.1` kèm Nuclio serverless; ban đầu thiếu gói `requirements-local.txt` (`numpy`) trên Python local và container `nuclio` thiếu newline trong store JSON, đã khắc phục.
- Dùng cache `data/cache/` thay vì tự chạy? <<không / có, vì ...>>
- Lỗi / bug script đã gặp (nếu có): <<>>
- Một điều bạn làm khác lần sau: <<>>
