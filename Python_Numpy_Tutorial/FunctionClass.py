print("---Python Functions---")
def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

for x in [-1, 0, 1]:
    print(sign(x))

def hello(name, loud=False):
    if loud:
        print('Hello, %s' % name.upper())
    else:
        print('Hello, %s' % name)

hello('Bob')
hello('Fred', loud=True)

print("---Python Classes")
class Greeter(object):

    #Constructor
    def __init__(self, name):
        self.name = name #Create an instance variable
    
    #Instance Method
    def greet(self, loud=False):
        if loud:
            print('HELLO, %s!' % self.name.upper())
        else:
            print('Hello, %s' % self.name)
#instance应该指的是类实例化的实例？
g = Greeter('Fred')
g.greet()
g.greet(loud=True)