import numpy as np
import random

class exGAN():
    def conditional_label(self, x_data, x_data_category, category_index, mask, cond=1):
        i = 0
        output_latlon = []
        output_day = []
        output_hour = []
        for category in x_data_category:
            temp_latlon = []
            temp_day = []
            temp_hour = []
            for k in range(len(x_data_category[i])-int(sum(mask[i])[0]), len(x_data_category[i])):
                pick = random.choice([True, True, True, True, True, True, True, True, True, True])  # 50% 

                if category[k][cond] == 1 and pick:
                    temp_latlon.append(category_index[cond][0])
                    temp_day.append(self.make_one_hot(category_index[cond][2], 7)) # aggregated information for the whole dataset, grouped by category.
                    temp_hour.append(self.make_one_hot(category_index[cond][3], 24))
                else:
                    temp_latlon.append(x_data[0][i][k])
                    temp_day.append(x_data[1][i][k])
                    temp_hour.append(x_data[2][i][k])
                    pass
                pass
            i += 1
            output_latlon.append(temp_latlon)
            output_day.append(temp_day)
            output_hour.append(temp_hour)
            pass

        

        output_latlon = np.array(output_latlon)
        output_day = np.array(output_day)
        output_hour = np.array(output_hour)
        return output_latlon, output_day, output_hour

    def make_one_hot(self, num, length):
        num_list = list(0 for i in range(length))
        num_list[num] = 1
        return num_list

    def average_value_by_category(self, x_data, mask):
        output_index = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
        each_category_latlon = list(0 for j in range(2))
        each_category_day = list(0 for j in range(7))
        each_category_hour = list(0 for j in range(24))
        for i in range(10):
            output_index[i].append(each_category_latlon)
            output_index[i].append([0])
            output_index[i].append(each_category_day)
            output_index[i].append(each_category_hour)
            pass

        x_data_latlon = x_data[0]
        x_data_day = x_data[1]
        x_data_hour = x_data[2]
        x_data_category = x_data[3]
        
        
        for i in range(len(x_data_category)):
            for j in range(len(x_data_category[i])-int(sum(mask[i])[0]), len(x_data_category[i])):    
                    
                    iter_index = x_data_category[i][j].tolist().index(1)

                    each_category_latlon = [output_index[iter_index][0][k] + x_data_latlon[i][j][k] for k in range(2)]
                    output_index[iter_index][0] = each_category_latlon
                    output_index[iter_index][1][0] += 1

                    each_category_day = [output_index[iter_index][2][k] + x_data_day[i][j][k] for k in range(7)]
                    output_index[iter_index][2] = each_category_day

                    each_category_hour = [output_index[iter_index][3][k] + x_data_hour[i][j][k] for k in range(24)]
                    output_index[iter_index][3] = each_category_hour
                    pass
            pass

        for k in range(10):
            
            if output_index[k][1][0] > 0:
                
                lat = output_index[k][0][0] / output_index[k][1][0]
                lon = output_index[k][0][1] / output_index[k][1][0]
                output_index[k][0] = [lat, lon]

                most_freq_day = output_index[k][2].index(max(output_index[k][2])) 
                most_freq_hour = output_index[k][3].index(max(output_index[k][3]))

                output_index[k][2] = most_freq_day
                output_index[k][3] = most_freq_hour
            else:
                continue



        return output_index