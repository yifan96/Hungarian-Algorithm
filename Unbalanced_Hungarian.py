import numpy as np


def min_zero_row(zero_mat, mark_zero):
    '''
    The function can be splitted into two steps:
    #1 The function is used to find the row which containing the fewest 0.
    #2 Select the zero number on the row, and then marked the element corresponding row and column as False
    '''

    # Find the row
    min_row = [99999, -1]

    for row_num in range(zero_mat.shape[0]):
        if np.sum(zero_mat[row_num] == True) > 0 and min_row[0] > np.sum(zero_mat[row_num] == True):
            min_row = [np.sum(zero_mat[row_num] == True), row_num]

    # Marked the specific row and column as False
    zero_index = np.where(zero_mat[min_row[1]] == True)[0][0]
    mark_zero.append((min_row[1], zero_index))
    zero_mat[min_row[1], :] = False
    zero_mat[:, zero_index] = False



def mark_matrix(mat):
    '''
    Finding the returning possible solutions for LAP problem.
    '''

    # Transform the matrix to boolean matrix(0 = True, others = False)
    cur_mat = mat
    zero_bool_mat = (cur_mat == 0)
    zero_bool_mat_copy = zero_bool_mat.copy()

    # Recording possible answer positions by marked_zero
    marked_zero = []
    while (True in zero_bool_mat_copy):
        min_zero_row(zero_bool_mat_copy, marked_zero)
    #print(marked_zero)

    # Recording the row and column positions seperately.
    marked_zero_row = []
    marked_zero_col = []

    for i in range(len(marked_zero)):
        marked_zero_row.append(marked_zero[i][0])
        marked_zero_col.append(marked_zero[i][1])
    print(marked_zero_row)
    print(marked_zero_col)

    # Step 2-2-1
    non_marked_row = list(set(range(cur_mat.shape[0])) - set(marked_zero_row))

    marked_cols = []
    check_switch = True
    while check_switch:
        check_switch = False
        for i in range(len(non_marked_row)):
            row_array = zero_bool_mat[non_marked_row[i], :]
            for j in range(row_array.shape[0]):
                # Step 2-2-2
                if row_array[j] == True and j not in marked_cols:
                    # Step 2-2-3
                    marked_cols.append(j)
                    check_switch = True

        for row_num, col_num in marked_zero:
            # Step 2-2-4
            if row_num not in non_marked_row and col_num in marked_cols:
                # Step 2-2-5
                non_marked_row.append(row_num)
                check_switch = True
    # Step 2-2-6
    marked_rows = list(set(range(mat.shape[0])) - set(non_marked_row))
    print(marked_rows)
    print(marked_cols)
    return (marked_zero, marked_rows, marked_cols)





def adjust_matrix(mat, cover_rows, cover_cols):
    cur_mat = mat
    non_zero_element = []

    # Step 4-1
    for row in range(len(cur_mat)):
        if row not in cover_rows:
            for i in range(len(cur_mat[row])):
                if i not in cover_cols:
                    non_zero_element.append(cur_mat[row][i])
    print(non_zero_element)
    min_num = min(non_zero_element)

    # Step 4-2
    for row in range(len(cur_mat)):
        if row not in cover_rows:
            for i in range(len(cur_mat[row])):
                if i not in cover_cols:
                    cur_mat[row, i] = cur_mat[row, i] - min_num
    # Step 4-3
    for row in range(len(cover_rows)):
        for col in range(len(cover_cols)):
            cur_mat[cover_rows[row], cover_cols[col]] = cur_mat[cover_rows[row], cover_cols[col]] + min_num

    return cur_mat

def unbalanced_assignment(mat):
    zero_mat = (mat == 0)
    #print(zero_mat)

    marked_zero = []
    zero_bool_mat_copy = zero_mat.copy()
    marked_zero_row = []
    marked_zero_col = []

    for i in range(len(marked_zero)):
        marked_zero_row.append(marked_zero[i][0])
        marked_zero_col.append(marked_zero[i][1])
    # Recording possible answer positions by marked_zero
    while len(marked_zero) < len(mat[0]):
        for row in range(len(zero_bool_mat_copy)):
            if np.sum(zero_bool_mat_copy[row] == True) == 1:
                marked_zero.append([row, np.where(zero_bool_mat_copy[row] == True)[0][0]])
                zero_bool_mat_copy[:, np.where(zero_bool_mat_copy[row] == True)[0][0]] = [False for i in range(
                    len(zero_bool_mat_copy))]
                # zero_bool_mat_copy[row, :] = [False for i in range(len(zero_bool_mat_copy[0]))]
                # break
                # print("change 1")
                # print(zero_bool_mat_copy)
                continue
                # if np.sum()
        # print(marked_zero)
        # Now the left
        left_zero_row = np.where(zero_bool_mat_copy == True)[0]
        left_zero_col = np.where(zero_bool_mat_copy == True)[1]
        # print(len(np.where(zero_bool_mat_copy==True)[0]))
        min_zero_index = [-1, -1]
        min_zero = np.Inf
        index = 0
        if len(marked_zero) < len(mat[0]):
            for i in range(len(np.where(zero_bool_mat_copy == True)[0])):
                if mat[left_zero_row[i]][left_zero_col[i]] < min_zero:
                    min_zero_index = [left_zero_row[i], left_zero_col[i]]
                    min_zero = mat[left_zero_row[i]][left_zero_col[i]]
                    index = i
            marked_zero.append(min_zero_index)
            if len(left_zero_col) > 0:
                zero_bool_mat_copy[:, left_zero_col[index]] = [False for i in range(len(zero_bool_mat_copy))]
            #print(zero_bool_mat_copy)
            #print(marked_zero)
    return marked_zero
def hungarian_algorithm(mat):
    #dim = max(mat.shape[0],mat.shape[1])
    dim = mat.shape[0]
    print(dim)
    cur_mat = mat

    # Step 1 - Every column and every row subtract its internal minimum

    for col_num in range(mat.shape[1]):
        cur_mat[:, col_num] = cur_mat[:, col_num] - np.min(cur_mat[:, col_num])
    for row_num in range(mat.shape[0]):
        cur_mat[row_num] = cur_mat[row_num] - np.min(cur_mat[row_num])
    print("substract col minima")
    print(cur_mat)


    print("substract row and colomn minima")
    print(cur_mat)
    zero_count = 0
    while zero_count < dim:
        # Step 2 & 3
        ans_pos, marked_rows, marked_cols = mark_matrix(cur_mat)
        zero_count = len(marked_rows) + len(marked_cols)

        if zero_count < dim:
            cur_mat = adjust_matrix(cur_mat, marked_rows, marked_cols)
        #print(cur_mat)
    print("cur_mat")
    print(cur_mat)
    result = unbalanced_assignment(cur_mat)
    print("ans pos")
    print(ans_pos)
    print(result)
    return ans_pos


def ans_calculation(mat, pos):
    total = 0
    ans_mat = np.zeros((mat.shape[0], mat.shape[1]))
    for i in range(len(pos)):
        total += mat[pos[i][0], pos[i][1]]
        ans_mat[pos[i][0], pos[i][1]] = mat[pos[i][0], pos[i][1]]
    return total, ans_mat


def main():
    '''Hungarian Algorithm:
    Finding the minimum value in linear assignment problem.
    Therefore, we can find the minimum value set in net matrix
    by using Hungarian Algorithm. In other words, the maximum value
    and elements set in cost matrix are available.'''

    cost_matrix = np.array([[300, 250, 180, 320, 270, 190, 220, 260],
                [290, 310, 190, 180, 210, 200, 300, 190],
                [280, 290, 300, 190, 190, 220, 230, 260],
                [290, 300, 190, 240, 250, 190, 180, 210],
                [210, 200, 180, 170, 160, 140, 160, 180]])

    ans_pos = hungarian_algorithm(cost_matrix.copy())  # Get the element position.
    print(ans_pos)
    ans, ans_mat = ans_calculation(cost_matrix, ans_pos)  # Get the minimum or maximum value and corresponding matrix.

    # Show the result
    print(f"Linear Assignment problem result: {ans:.0f}\n{ans_mat}")

    # # If you want to find the maximum value, using the code as follows:
    # # Using maximum value in the cost_matrix and cost_matrix to get net_matrix
    # profit_matrix = np.array([[7, 6, 2, 9, 2],
    #                           [6, 2, 1, 3, 9],
    #                           [5, 6, 8, 9, 5],
    #                           [6, 8, 5, 8, 6],
    #                           [9, 5, 6, 4, 7]])
    # max_value = np.max(profit_matrix)
    # cost_matrix = max_value - profit_matrix
    # ans_pos = hungarian_algorithm(cost_matrix.copy())  # Get the element position.
    # ans, ans_mat = ans_calculation(profit_matrix, ans_pos)  # Get the minimum or maximum value and corresponding matrix.
    # # Show the result
    # print(f"Linear Assignment problem result: {ans:.0f}\n{ans_mat}")


if __name__ == '__main__':
    main()

#    [[0, 2], [2, 3], [3, 6], [1, 7], [4, 0], [4, 1], [4, 4], [4, 5]]