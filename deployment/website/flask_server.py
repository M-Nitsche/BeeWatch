from flask import Flask, render_template, redirect, url_for
from flask_bootstrap import Bootstrap

app = Flask(__name__)
bootstrap = Bootstrap(app)

@app.route('/')
def demo():
    return render_template('demo.html')

@app.route('/models')
def models():
    return render_template('models.html')

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

if __name__ == '__main__':
    #app.run(debug=True, ssl_context='adhoc') # ssl_context='adhoc' for https https://blog.miguelgrinberg.com/post/running-your-flask-application-over-https
    app.run(debug=True, port=5000)



















