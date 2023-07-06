from django.shortcuts import render
from django.views import View
from django.core.files.storage import FileSystemStorage
from .image_recognition import ImageRecognitionModel
import numpy as np
from PIL import Image

class ImageUploadView(View):
    def get(self, request):
        return render(request, 'upload.html')

    def post(self, request):
        uploaded_image = request.FILES['image']
        fs = FileSystemStorage()
        image_path = fs.save(uploaded_image.name, uploaded_image)
        image_url = fs.url(image_path)
        ai_model = ImageRecognitionModel()

        # Open and resize the uploaded image
        image = Image.open(uploaded_image)
        image = image.resize((224, 224))  # Resize to the desired input shape
        image_array = np.array(image)

        predictions = ai_model.predict_image(image_array)
        return render(request, 'upload.html', {'image_url': image_url, 'predictions': predictions})