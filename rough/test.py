#test Area
import os

def main():
    print('Always run !')
    with open(os.path.join("rough","test.txt"),"w") as d:
        d.write("Garbage File !!")



if __name__ == '__main__':
    main()

