from flask import Flask
app = Flask(__name__)

@app.route('/')
def indext_get():
    return "hello"
