# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 429; đối tượng sau duyệt: 296; ngưỡng accepted: IoU >= 0.9
- Đã **loại 19 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 279 | 279/429 = 65.0% | 429 box máy |
| edited (sửa) | 10 | 10/429 = 2.3% | 429 box máy |
| deleted (xoá) | 129 | 129/429 = 30.1% | 429 box máy |
| added (vẽ thêm) | 7 | 7/296 = 2.4% | 296 đối tượng sau duyệt |

Trong số deleted + added, **4 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 3.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 137 | 1 | 79 | 1 |
| bicycle | 1 | 3 | 2 | 0 |
| motorbike | 97 | 2 | 26 | 2 |
| car | 31 | 0 | 7 | 2 |
| bus | 2 | 0 | 1 | 0 |
| truck | 0 | 0 | 5 | 0 |
| three_wheeler | 11 | 4 | 9 | 2 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 20 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 18 | 37 | 2 | 32.7% (KTC 95%: 22–46%) | 90.0% (KTC 95%: 70–97%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 9/12 = 75.0% (KTC 95%: 47–91%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - loose: 3/3
  - extra: 3/3
  - wrong_class: 2/3
  - missing: 1/3
  - lọt: img_029.jpg (wrong_class), img_005.jpg (missing), img_004.jpg (missing)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
