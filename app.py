from flask import Flask
import model

app = Flask(__name__)

'''
    
    this is for flask service
    
    1. post api for get problem_set
    
    
    
'''

print("hello world!")

@app.route('/')
def hello_world() :
    return 'hello world!'


if __name__ == "__main__" :
    app.run()

