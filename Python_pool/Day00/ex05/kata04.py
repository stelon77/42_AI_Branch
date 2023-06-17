#!/usr/bin/env python3
t = (0, 4, 132.42222, 10000, 12345.67)

print(('module_{:02d}, ex_{:02d} : {:.2f}, {:.2e}, {:.2e}'
      .format(t[0], t[1], t[2], t[3], t[4])))
