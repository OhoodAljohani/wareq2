# STEP 1
# import libraries
from posixpath import dirname
import fitz
import io
from PIL import Image
  
# STEP 2
# file path you want to extract images from
file = "data/Riyadh-Plants-Manual-images2_organized.pdf"
  
# open the file
pdf_file = fitz.open(file)
i = 0
import os
# STEP 3
# iterate over PDF pages
# iterate over pdf pages
for page_index in range(len(pdf_file)):
    # get the page itself
    page = pdf_file[page_index]
    image_list = page.getImageList()
    # printing number of images found in this page
    if image_list:
        print(f"[+] Found a total of {len(image_list)} images in page {page_index}")
        dirname = "./data/images/"+str(page_index)
        os.makedirs(dirname)   
    else:
        print("[!] No images found on page", page_index)
    for image_index, img in enumerate(page.getImageList(), start=1):
        # get the XREF of the image
        xref = img[0]
        # extract the image bytes
        base_image = pdf_file.extractImage(xref)
        image_bytes = base_image["image"]
        # get the image extension
        image_ext = base_image["ext"]
        # load it to PIL
        image = Image.open(io.BytesIO(image_bytes))
        # save it to local disk
        for i in range(len(image_list)):
            image.save(open(f"{dirname}/image{page_index+1}_{image_index}.{image_ext}", "wb"))
image_bytes
from PIL import Image
import io

image_data = image_bytes
image = Image.open(io.BytesIO(image_data))
image.show()

