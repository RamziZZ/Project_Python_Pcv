from PIL import Image

# Load image
image_path = 'your_image_path_here.jpg'  # Ganti dengan path gambar Anda
image = Image.open(image_path)

# Definisikan bounding box (kiri, atas, kanan, bawah)
left = 50
top = 50
right = 350
bottom = 250

# Crop gambar
cropped_image = image.crop((left, top, right, bottom))

# Simpan hasil cropping
cropped_image.save('cropped_image.jpg')

# Tampilkan gambar hasil cropping
cropped_image.show()
