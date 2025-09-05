import cv2
import numpy as np

# upscale_image 함수
def upscale_image(image, target_width=2500):
    height, width = image.shape[:2]
    if width < target_width:
        target_height = int((target_width / width) * height)
        image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        print(f"이미지 업스케일링: ({width}, {height}) -> ({target_width}, {target_height})")
    return image

# 1. 프론트엔드로부터 이미지와 네 꼭짓점 좌표를 전달받았다고 가정
image_path = 'test_image2.jpg'
image = cv2.imread(image_path)

# 프론트엔드에서 사용자가 조정한 네 좌표 (좌상단, 우상단, 우하단, 좌하단 순서)
# 이 값은 실제로는 HTTP 요청 등으로 전달받아야 함
four_corner_coords_from_frontend = np.array([
    [361, 25],   # 좌상단 (예시 좌표)
    [3801, 9],  # 우상단 (예시 좌표)
    [4001, 5057], # 우하단 (예시 좌표)
    [265, 5081]   # 좌하단 (예시 좌표)
], dtype="float32")

if image is None:
    print(f"'{image_path}' 파일을 열 수 없습니다.")
else:
    # 2. 전달받은 좌표로 원근 변환 실행
    rect = four_corner_coords_from_frontend
    (tl, tr, br, bl) = rect

    # 너비와 높이 계산
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # 최종 결과물이 될 반듯한 사각형의 좌표
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 변환 행렬을 계산하고 적용
    M = cv2.getPerspectiveTransform(rect, dst)
    warped_image = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    # 3. 펴진 이미지에 업스케일 적용
    upscaled = upscale_image(warped_image, target_width=2500)

    # 4. 그레이 스케일 변환
    gray_image = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

    # 5. 노이즈 제거
    denoised = cv2.medianBlur(gray_image, 3)

    # 6. 이진화
    binary_image = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 4
    )
    
    # 7. 처리된 이미지 저장
    processed_image_path = 'processed_image.png'
    cv2.imwrite(processed_image_path, binary_image)
    print(f"이미지 전처리 완료. '{processed_image_path}' 파일 생성 완.")