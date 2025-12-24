import cv2
import numpy as np
import os

# ===================== パス設定 =====================
img_path = r"C:\創造演習\元画像_着色済み\1.png"
save_dir = r"C:\Users\taoso\研究\創造演習\出力画像3"

# ===================== パラメータ =====================
display_scale = 0.25
S_tol = 12
V_tol = 10
scan_plane = np.ones((5, 5), np.uint8)
margin = 2

# ===================== 画像読み込み =====================
path = np.fromfile(img_path, np.uint8)
img = cv2.imdecode(path, cv2.IMREAD_COLOR)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

h_img, w_img, _ = img.shape
display_img = cv2.resize(img, None, fx=display_scale, fy=display_scale)

# ===================== グローバル変数 =====================
binary_result = None
binary_display = None
contours = []

click_count = 0
last_rect = None
cut_count = 0

# ===================== 二値化 =====================
def on_mouse_original(event, x, y, flags, param):
    global binary_result, binary_display, contours

    if event == cv2.EVENT_LBUTTONDOWN:
        orig_x = int(x / display_scale)
        orig_y = int(y / display_scale)

        if orig_x < 0 or orig_y < 0 or orig_x >= w_img or orig_y >= h_img:
            return

        clicked_s = int(s[orig_y, orig_x])
        clicked_v = int(v[orig_y, orig_x])

        print(f"二値化クリック: ({orig_x}, {orig_y})  S={clicked_s}, V={clicked_v}")

        sv_sim_map = np.where(
            (np.abs(s.astype(np.int16) - clicked_s) <= S_tol) &
            (np.abs(v.astype(np.int16) - clicked_v) <= V_tol),
            255, 0
        ).astype(np.uint8)

        sv_sim_map = cv2.morphologyEx(sv_sim_map, cv2.MORPH_OPEN, scan_plane)
        sv_sim_map = cv2.morphologyEx(sv_sim_map, cv2.MORPH_CLOSE, scan_plane)

        binary_result = sv_sim_map

        contours, _ = cv2.findContours(
            binary_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        binary_display = cv2.resize(
            binary_result, None,
            fx=display_scale, fy=display_scale,
            interpolation=cv2.INTER_NEAREST
        )

        cv2.imshow("binary", binary_display)

# ===================== 矩形表示＆切り出し =====================
def on_mouse_binary(event, x, y, flags, param):
    global click_count, last_rect, cut_count

    if binary_result is None:
        return

    orig_x = int(x / display_scale)
    orig_y = int(y / display_scale)

    if event == cv2.EVENT_LBUTTONDOWN:
        if binary_result[orig_y, orig_x] == 255:
            for contour in contours:
                if cv2.pointPolygonTest(contour, (orig_x, orig_y), False) >= 0:
                    x1, y1, w, h = cv2.boundingRect(contour)

                    display = img.copy()
                    cv2.rectangle(
                        display,
                        (x1 - margin, y1 - margin),
                        (x1 + w + margin, y1 + h + margin),
                        (0, 0, 255), 2
                    )

                    cv2.imshow("selection", cv2.resize(
                        display, None, fx=display_scale, fy=display_scale
                    ))

                    if last_rect == (x1, y1, w, h):
                        click_count += 1
                    else:
                        click_count = 1
                        last_rect = (x1, y1, w, h)

                    if click_count == 2:
                        cut_count += 1

                        xs = max(x1 - margin, 0)
                        ys = max(y1 - margin, 0)
                        xe = min(x1 + w + margin, img.shape[1])
                        ye = min(y1 + h + margin, img.shape[0])

                        crop = img[ys:ye, xs:xe]
                        save_path = os.path.join(save_dir, f"cutout_{cut_count}.png")

                        ret, buf = cv2.imencode(".png", crop)
                        if ret:
                            buf.tofile(save_path)
                            print("保存:", save_path)

                        click_count = 0
                    break

# ===================== ウィンドウ設定 =====================
cv2.namedWindow("original")
cv2.namedWindow("binary")
cv2.namedWindow("selection")

cv2.imshow("original", display_img)

cv2.setMouseCallback("original", on_mouse_original)
cv2.setMouseCallback("binary", on_mouse_binary)

# ===================== ループ =====================
while True:
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()