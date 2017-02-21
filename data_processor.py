__author__ = 'Air'
import matplotlib.pyplot as plt

afile = open("/home/air/data_log.txt")
line = afile.readline()
count = 0
success_session = []
average_step = []
average_award = []

while line:
    #process
    elements = line.split(';')
    assert len(elements) == 3
    success_session.append(int(elements[0]))
    average_step.append(int(elements[1]))
    average_award.append(float(elements[2]))

    line = afile.readline()
    count += 1

plt.figure(1)
plt.plot(average_step)
plt.figure(2)
plt.plot(average_award)
# plt.show()

AVERGE_STEP = 20
success_percent = []
fail_percent = []
out_percent = []
for i in range(0, len(success_session)/AVERGE_STEP - 1):
    data_cut = success_session[i * AVERGE_STEP :(i+1) * AVERGE_STEP - 1]
    session_cout = [0, 0, 0]
    for data in data_cut:
        session_cout[int(data)] += 1
    success_percent.append(session_cout[2]/float(AVERGE_STEP))
    fail_percent.append(session_cout[0]/float(AVERGE_STEP))
    out_percent.append(session_cout[1]/float(AVERGE_STEP))

print len(success_percent)*AVERGE_STEP
plt.figure(3)
plt.plot(success_percent)
plt.plot(fail_percent)
plt.plot(out_percent)
plt.show()



print count
