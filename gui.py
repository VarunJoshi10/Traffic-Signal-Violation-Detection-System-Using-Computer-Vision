from tkinter import *
from tkinter import filedialog, ttk
from tkinter import messagebox
import time
from traffic import violation
from tripling import detect
def button_exit():
    MsgBox = messagebox.askquestion(
        'Exit Application', 'Are you sure you want to exit the application', icon='warning')
    if MsgBox == 'yes':
        root.destroy()
    else:
        messagebox.showinfo(
            'Return', 'You will now return to the application screen')
    return

#GUI

root =Tk()
root.title("Violation System")
root.geometry('650x400')

im= PhotoImage(file="background.png")
logoImage=Label(root,image=im)
logoImage.place(x=0,y=0)

jai=Label(root,text="Traffic Signal Violation Detection System",fg="red",font=("Arial",20,"bold"))
jai.place(x=75,y=70)

button_detection=Button(root,text="Red Light Violation",padx=25,pady=10,command=violation,bg='green',fg='white')
button_detection.place(x=225,y=200)

trip_button_detection=Button(root,text="Tripling Violation",padx=30,pady=10,command=detect,bg='green',fg='white')
trip_button_detection.place(x=225,y=250)


button_exit = Button(
    root, text="EXIT",
    padx=60, pady=10,
    bg='red', fg="white", command=button_exit
)
button_exit.place(x=225,
                  y=300)
develop=Label(root,text="Developed by: Group No 11",bg="black",fg="white",font=("Arial",15,"bold"))
develop.place(x=200,y=360)
root.mainloop()