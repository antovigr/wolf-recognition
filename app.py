from flask import Flask,render_template,url_for,request,flash, redirect
from werkzeug.utils import secure_filename

from model1 import *

app = Flask(__name__)

UPLOAD_FOLDER = r'D:\Py_projects\animal\static\img'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "super secret key"

model=getModel()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/prediction',methods=['POST','GET'])
def prediction():

    if 'picture' not in request.files:

        flash('No file part')
        return redirect(request.url)
    file = request.files['picture']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print(filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    img = cv2.imread(app.config['UPLOAD_FOLDER'] + "\\" + filename)

    if not(img is None):
        resized_img = cv2.resize(img, (224, 224)) #MobileNetv2 model
        resized_img = resized_img/255
        resized_img = np.expand_dims(resized_img, axis=0)

        my_prediction = model.predict(resized_img)

        print(my_prediction)

        if abs(my_prediction[0][1]) > abs(my_prediction[0][0]):
            pred=0
        else:
            pred=1

        return render_template('prediction.html', tab = [pred, filename])

    return redirect("localhost:5000/home.html")


