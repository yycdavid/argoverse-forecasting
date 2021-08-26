step = 0

def main():
    step = 2
    foo() 
    foo()
    
def foo():
    global step 
    step += 1 
    print(step)

main()