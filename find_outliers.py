import numpy as np

def find_outliers(points, sigma_dist):
    avg = np.average(points)
    std = np.std(points)
    return (list(points[points > avg + sigma_dist * std]) +
        list(points[points < avg - sigma_dist * std]))

if __name__=='__main__':
    for i in range(100):
        points = list(np.random.randn(100))
        outliers = find_outliers(points, 2.0)
        print len(outliers)