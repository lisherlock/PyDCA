''' Decision Curve Analysis，DCA '''
def DCA_curve_show(Output_folder, model_type_temp):
    print('Decision Curve Analysis is running... ')

    model_name_list = model_type_temp.split('  ')
    list.pop(model_name_list)

    model_pre = dict()
    for i in range(len(model_name_list)):
        data_temp= pd.read_csv(os.path.join(Output_folder, 'Models_Result/Model_pre_save/' + model_name_list[i] + '_test_model_Pre.csv'))
        model_pre[i] = data_temp['label_1_pre'].values


    y_true = data_temp['label'].values


    def model_thre(pred_proba, thres):
        new_label = []
        for i in range(len(pred_proba)):
            if pred_proba[i] > thres:
                new_label.append(1)
            else:
                new_label.append(0)
        new_label = np.array(new_label)
        return new_label

 
    def calculate_DCA_curve(y, x):
        return trapz(y, x, dx=0.001)


    def pyDCA(pred_proba, y_true):
        Pt_x = []
        NB_y = []
        NB_large_y = []
        NB_min_y = []

        for Pt in (np.arange(0.0, 1.0, 0.001)):
            pred_prediction = model_thre(pred_proba, Pt)
            confmat = confusion_matrix(y_true=y_true, y_pred=pred_prediction) 

            A = (confmat[1, 1]) / len(y_true)
            B = (confmat[0, 1]) / len(y_true)
            C = (confmat[1, 0]) / len(y_true)
            D = (confmat[0, 0]) / len(y_true) 

            NB = A - B * Pt / (1 - Pt)
            #        NB = A*(A+C) - B*(1-A-C) * Pt / (1 - Pt) 

        
        
            num_0 = 0
            num_1 = 0
            for i in range(len(y_true)):
                if y_true[i] == 0:
                    num_0 = num_0 + 1
                elif y_true[i] == 1:
                    num_1 = num_1 + 1

            A_all = num_1 / (num_0 + num_1)
            NB_large = A_all - (1 - A_all) * Pt / (1 - Pt) 


           
            Pt_x.append(Pt)
            NB_y.append(NB)
            NB_large_y.append(NB_large)
            NB_min_y.append(0)

        return Pt_x, NB_y, NB_large_y, NB_min_y


    xmin, xmax = 0, 1.0
    ymin, ymax = -0.1, 1.0
    plt.figure(figsize=(8, 6), dpi=100)
    curve_color = ['dodgerblue', 'deeppink', 'gold', 'turquoise', 'lime', 'orangered', 'rosybrown', 'darkturquoise', 'blueviolet', 'lime', 'yellow',
                   'darkorange', 'crimson', 'pink', 'hotpink', 'violet', 'purple', 'teal', 'paleturquoise']

    for i in range(len(model_name_list)):
        Pt_x, NB_y, NB_large_y, NB_min_y = pyDCA(model_pre[i], y_true)
        plt.plot(Pt_x, NB_y, linestyle='-', lw=1.8, alpha=1, color=curve_color[i],
                 label= (model_name_list[i] +  ' ' + '%0.3f' % calculate_DCA_curve(NB_y, Pt_x)))

    plt.plot(Pt_x, NB_large_y, linestyle='--', lw=2, alpha=1, color='gray', label='All positive')
    plt.plot(Pt_x, NB_min_y, linestyle='--', lw=1.5, alpha=1, color='black', label='All negative')

    plt.title("Decision Curve Analysis")
    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False 
    ax = plt.axes()
    ax.spines['top'].set_visible(False) 
    ax.spines['right'].set_visible(False) 
    plt.legend()
    plt.xlabel('High Risk Threshold')
    plt.ylabel('Net Benefit')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.grid(False) 
    plt.legend(loc='upper right', frameon=True)
    plt.savefig(os.path.join(Output_folder, 'Models_Result/' + 'Decision Curve Analysis of Model.png'))
