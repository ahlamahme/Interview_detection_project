#import win32api
#from flask import Flask, render_template

#app = Flask(__name__)


#Using the below, the popup message appears on the page load of index.html
#0x00001000 - This makes the popup appear over the browser window
#@app.route('/t')
#def index():
 #   win32api.MessageBox(0, 'You have just run a python script on the page load!', 'Running a Python Script via Javascript', 0x00001000)
  #  return render_template('session.html')



#if __name__ == "__main__":
 #   app.run(debug=True)
from werkzeug.middleware.dispatcher import DispatcherMiddleware # use to combine each Flask app into a larger one that is dispatched based on prefix
from facial import app as flask_app_1
from facial import app1 as flask_app_2
from werkzeug.serving import run_simple # werkzeug development server
application = DispatcherMiddleware(flask_app_2, {
    '/flask_app_2': flask_app_1
})

if __name__ == '__main__':
    run_simple('localhost', 5000, application, use_reloader=True, use_debugger=True, use_evalex=True)