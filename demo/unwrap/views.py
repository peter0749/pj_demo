from django.shortcuts import render
from unwrap.models import IMG
from django.db import transaction
from django.http import StreamingHttpResponse
def upload(request):
    return render(request, 'unwrap/upload.html')
def mobile(request):
    return render(request, 'unwrap/mobile.html')
def show(request):
    try:
        with transaction.atomic():
            new_img = IMG(img=request.FILES.get('img'))
            new_img.save()
            content = {
                'aaa': new_img.url_path,
                'bbb': new_img.wrap_url_path,
                'aaa_name': new_img.original_filename,
                'bbb_name': new_img.download_filename
            }
    except:
        return render(request, 'unwrap/error.html')
    return render(request, 'unwrap/show.html', content)
def download(request):
    try:
        with transaction.atomic():
            new_img = IMG(img=request.FILES.get('img'))
            new_img.save()
            response = StreamingHttpResponse(readFile(new_img.wrap_path))
            response['Content-Type']='application/octet-stream'
            response['Content-Disposition']='attachment;filename="%s"'%new_img.download_filename
    except:
        return render(request, 'unwrap/error.html')
    return response

def readFile(filename,chunk_size=512):
    with open(filename,'rb') as f:
        while True:
            c=f.read(chunk_size)
            if c:
                yield c
            else:
                break
