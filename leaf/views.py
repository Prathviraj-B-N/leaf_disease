from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

import os
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from leaf import predict
import shutil
import os

@csrf_exempt
# Create your views here.
def home(request):

    if request.method == "GET":
        return render(request, 'index.html')

    if request.method == "POST":
        image = request.FILES['image']
        print(os.getcwd())
        if os.path.exists(os.getcwd() + '\\media') and os.path.isdir(os.getcwd() + '\\media'):
            shutil.rmtree(os.getcwd() + '\\media')

        path = default_storage.save(os.getcwd() +'\\media\\result.jpg', ContentFile(image.read()))
        tmp_file = os.path.join(settings.MEDIA_ROOT, path)

        result = predict.process()

        print(result)

    return render(request, "result.html",{'result':result})
