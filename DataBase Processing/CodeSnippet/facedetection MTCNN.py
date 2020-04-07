# from mtcnn import MTCNN                           # for face detection
# from matplotlib import pyplot                     # for face detection
# from matplotlib.patches import Rectangle          # for face detection

# mtcnn = MTCNN()

# for face detection
def detect_Face(img_path):

    img = pyplot.imread(img_path)
    faces = mtcnn.detect_faces(img)
    for face in faces:
        print(face)

    draw_image_with_boxes(img, faces)

# draw an image with detected objects
def draw_image_with_boxes(img, faces):
    # plot the image
    pyplot.imshow(img)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    for face in faces:
        # get coordinates
        x, y, width, height = face['box']
        # create the shape
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        # draw the box
        ax.add_patch(rect)
    # show the plot
    pyplot.show()