# Thông tin học viên

- Họ và tên: Lê Đức Lương
- MSHV: 2A202603137
- Lớp (L2 / L3 / khác): L23
- Tài khoản GitHub: haryluong1
- Repo bài nộp (URL): https://github.com/haryluong1/K5-L23-Day08-LeDucLuong-2A202603137-AI-Assisted-Labeling
- Commit chốt bài (hash): 2347323 (cập nhật commit cuối khi push)
- Máy: Windows 11, CPU (không có GPU rời)
- CVAT (Docker): v2.74.1

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: 279/429 = 65.0% / 10/429 = 2.3% / 129/429 = 30.1% (mẫu số: 429 box máy sau khi trừ vùng bẫy); added: 7/296 = 2.4% (mẫu số: 296 đối tượng sau duyệt; trong đó 4 ca đổi lớp, máy sót thật = 3)
- So với gold: precision / recall của **máy**: Precision = 32.7% (18/55 box máy trên gold, KTC 95%: 22–46%) / Recall = 90.0% (18/20 đối tượng gold, KTC 95%: 70–97%)
- Bắt bẫy: 9/12 = 75.0% (loose: 3/3, extra: 3/3, wrong_class: 2/3, missing: 1/3)
- Thời gian: gán tay 3.75 phút/ảnh (15 phút / 4 ảnh gold); duyệt có pre-label 1.88 phút/ảnh (45 phút / 24 ảnh batch)
- Lỗi QC: 80.0% theo ảnh (4/5 ảnh mẫu QC, cận trên Wilson 95% = 96%), 7.3% theo đối tượng (4/55 đối tượng trong mẫu QC)
- **Quyết định:** Rework — lý do một dòng: Tỉ lệ ảnh có lỗi trong mẫu QC là 4/5 = 80.0% (> ngưỡng 2%) và tỉ lệ bắt bẫy đạt 9/12 = 75.0% (< ngưỡng 80%), cần sửa các lỗi biên/trùng box và duyệt bù 3 ảnh lọt bẫy theo Guideline v1.1 trước khi Release.
