import MySQLdb
import wx

'''
查询所有数据库中的密码和账户
'''

def find_passwd(db_name):
    try:
        my_db = MySQLdb.connect(host="localhost", user='root', passwd='5201314', database=db_name)
        my_cursor = my_db.cursor()

        if db_name == my_db_list[0]:
            my_cursor.execute('SELECT Account_Name, Account_PSW FROM account')
            my_result = my_cursor.fetchall()
            my_db.close()
            return db_name, my_result[0], my_result[1]

        elif db_name == my_db_list[1]:
            my_cursor.execute('SELECT Power,PSW FROM account')
            my_result = my_cursor.fetchall()
            my_db.close()
            return db_name, my_result[0], my_result[1]

        elif db_name == my_db_list[2]:
            my_cursor.execute("SELECT setting_key,setting_value FROM setting WHERE setting_key = 'BossPsw' OR setting_key = 'EmployeePsw'")
            my_result = my_cursor.fetchall()
            my_db.close()
            return db_name, my_result[0], my_result[1]
        elif db_name == my_db_list[3] or db_name == my_db_list[4]:
            my_cursor.execute("SELECT setting_name,setting_value FROM setting WHERE setting_name = 'Boss_psw' OR setting_name = 'Employee_psw'")
            my_result = my_cursor.fetchall()
            my_db.close()
            return db_name, my_result[0], my_result[1]
        else:
            print('error not find database')
    except Exception as e:
        print(e)
        return None


passwd_str = ""
my_db_list = ['eggchair', 'gameplatform', 'haiyang_park', 'rotation', 'warfare']
server = MySQLdb.connect(host="localhost", user='root', passwd='5201314')
c = server.cursor()
c.execute('SHOW DATABASES')
db_list = c.fetchall()
server.close()
db_list = [i[0] for i in db_list]
for db in db_list:
    if db in my_db_list:
        passwd_str += str(find_passwd(str(db))) + '\n' + '\n'


if __name__ == '__main__':
    app = wx.App()

    frame = wx.Frame(None, title='Find User Password')
    frame.SetSize(600, 400)

    panel = wx.Panel(frame)
    box = wx.BoxSizer(wx.VERTICAL)
    lb1 = wx.StaticText(panel, -1, style=wx.ALIGN_CENTER)
    font = wx.Font(18, wx.ROMAN, wx.ITALIC, wx.NORMAL)
    lb1.SetFont(font)
    if len(passwd_str) > 5:
        lb1.SetLabel(passwd_str)
    else:
        lb1.SetLabel('Not Find')
    box.Add(lb1, 0, wx.ALIGN_LEFT)
    panel.SetSizer(box)

    icon = wx.Icon('lcdz.ico', wx.BITMAP_TYPE_ICO)
    frame.SetIcon(icon)
    frame.Center()
    frame.Show()
    app.MainLoop()






