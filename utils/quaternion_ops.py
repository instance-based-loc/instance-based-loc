import numpy as np

class QuaternionOps:
    @staticmethod
    def quaternion_multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([w, x, y, z])

    @staticmethod
    def quaternion_conjugate(q):
        w, x, y, z = q
        return np.array([w, -x, -y, -z])

    # https://math.stackexchange.com/a/3573308
    @staticmethod
    def quaternion_error(q1, q2): # returns orientation angle between the two
        q_del = QuaternionOps.quaternion_multiply(QuaternionOps.quaternion_conjugate(q1), q2)
        q_del_other_way = QuaternionOps.quaternion_multiply(QuaternionOps.quaternion_conjugate(q1), -q2)
        return min(np.abs(np.arctan2(np.linalg.norm(q_del[1:]), q_del[0])),
                   np.abs(np.arctan2(np.linalg.norm(q_del_other_way[1:]), q_del_other_way[0])))