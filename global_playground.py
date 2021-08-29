step = 0

def main():
    global step
    foo() 
    step += 1
    foo()
    
def foo():
    global step 
    step += 1 
    print(step)

main()