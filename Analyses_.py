# First importing necessary packages
import numpy as np
import os
import time
import matplotlib.pyplot as plt 
import seaborn as sns
import h5py as h5
from scipy import interpolate
import pandas as pd
import pdb
import warnings
# Just read files in h5 format





class ETL:
	# class of loading data and preparing as array
    def __init__(self, sf, ending, unit_length):
        self.sf = sf # sampling frequency
        self.ending = ending
        self.unit_length = unit_length # in second

    def get_path(self, main_path):
        self.main_path = main_path
        print(f'Your entered path is {main_path}')

        temp = os.listdir(main_path)
        all_files = [file_ for file_ in temp if file_.endswith(self.ending)]
        self.all_files = all_files
        self.nr_files = np.array(len(all_files), dtype = np.int16)
        print('All files in the given path \n')
        print(all_files, f' Number of all files {len(all_files)}')

        
    def load_files(self):
        main_path = self.main_path
        all_files = self.all_files
        all_data = [h5.File(os.path.join(main_path, file_),'r') for file_ in all_files] # F in h5py in a list format for all data
        self.all_data = all_data
        print('Data are loaded \n')
        print('Start reading units \n')
        
        unit_length = self.unit_length
        sf = self.sf
        data_length = np.array(unit_length * sf, dtype=np.int16)
        nr_cells = np.array(all_data[0]['Amp'].shape[0], dtype=np.int16)
        self.data_length = data_length
        self.nr_cells = nr_cells
        print(f'unit length is {unit_length} sec, data length is {data_length} samples, number of cells are {nr_cells} \n')

        values_ = np.zeros((data_length, nr_cells, len(all_data)))

        for i in range(self.nr_files):
            for ii in range(nr_cells):
                values_[np.array(np.array(all_data[i][all_data[i]['Location'][ii][0]]), dtype = np.int64), ii, i] = np.array(all_data[i][all_data[i]['Amp'][ii][0]], dtype = np.float32)

        print('Finished reading units \n')
        print('Start reading hypno files \n')
        
        temp = [np.array(F['hypno'][0], dtype= np.int16) for F in all_data]
        temp = np.vstack(temp).T
        hypno = np.zeros((data_length, self.nr_files))

        for i in range(self.nr_files):
            f = interpolate.interp1d(np.linspace(0,temp.shape[0],temp.shape[0]), temp[:,i], kind='nearest')
            hypno[:,i] = f(np.linspace(0,temp.shape[0],data_length))

        print('Finished reading hypno \n')
        

        # initializing artifacts for hypno and data
        data_artifacts = np.zeros_like(values_) 
        hypno_artifacts = np.zeros_like(hypno) 

        self.data_artifacts = data_artifacts
        self.hypno_artifacts = hypno_artifacts
        self.all_data_values = values_
        self.hypno = hypno


    def get_data_values(self):
        print('Location-Time information')
        return self.all_data_values

    def get_hypno(self):
        return self.hypno

    def get_artifacts(self):
        print('output = data_artifacts, hypno_artifacts')
        return self.data_artifacts, self.hypno_artifacts

    def set_artifact(self, start_, end_, file_):
        print('Artifact indices will be applied on Location, Amplitude and Hypno files \n')
        data_artifacts = self.data_artifacts
        hypno_artifacts = self.hypno_artifacts
        print(data_artifacts.shape)
        hypno_artifacts[start_: end_, file_] = 1
        data_artifacts[start_:end_,:,file_] = 1
        self.data_artifacts = data_artifacts
        self.hypno_artifacts = hypno_artifacts

    


class activation:
    def __init__(self, hypno, data, hypno_artifacts, data_artifacts):
        # I consider hypno and data and artifacts are already prepared (if necessary for stacking)
        self.hypno = hypno
        self.data = data
        self.hypno_artifacts = hypno_artifacts
        self.data_artifacts = data_artifacts

    def mean_fire(self, file_nr, sr, fr_zeros):
        """
        mean firing rate is average of events normalized to length of input data per each individual cell
        if fr_zeros = False then zeros in mean fr analyses become nan
        """
        hypno = self.hypno
        data = self.data
        hypno_artifacts = self.hypno_artifacts
        data_artifacts = self.data_artifacts
        
        if  hasattr(data_artifacts, '__len__') and hasattr(hypno_artifacts, ' __len__'):
            data = data[:,:,file_nr]
            hypno = hypno[:,file_nr]
            data_artifacts = data_artifacts[:,:,file_nr]
            hypno_artifacts = np.squeeze(hypno_artifacts[:,file_nr])
            
            # it is necessary for normalization
            artifacts_length_sws = len([a for a in range(data.shape[0]) if (hypno[a] == -2 and hypno_artifacts[a] == 1)])
            artifacts_length_rem = len([a for a in range(data.shape[0]) if (hypno[a] == -3 and hypno_artifacts[a] == 1)])
            artifacts_length_awake = len([a for a in range(data.shape[0]) if (hypno[a] == 0 and hypno_artifacts[a] == 1)])
        else:
            data = data[:,:,file_nr]
            hypno = hypno[:,file_nr]
            artifacts_length_sws = 0
            artifacts_length_rem = 0
            artifacts_length_awake = 0
            
        
        if hasattr(data_artifacts, '__len__') and hasattr(hypno_artifacts, ' __len__'):
            data = data * np.logical_not(data_artifacts)    
            hypno = hypno * np.logical_not(hypno_artifacts)

        

        # check each state existance then # calculating mean firing rate & # claculating mean amplitude
        if not(np.isnan(np.where(hypno == -2 , 1, np.nan)).all()):
            data_sws = [np.where(np.where(hypno == -2 , 1, 0), data[:,i], 0) for i in range(data.shape[1])]
            data_sws = np.array(data_sws).T
            mean_fr_sws = np.sum(np.where(data_sws > 0 , 1, 0), axis = 0) / ((np.sum(np.where(hypno == -2 , 
                                    1, 0)) - artifacts_length_sws) / sr)
            mean_amp_sws = np.sum(np.where(data_sws > 0, data_sws, 0), 
                                    axis = 0) / np.sum(np.where(data_sws > 0, 1, 0), axis = 0)
        else:
            data_sws = [np.where(np.where(hypno == -2 , 1, 0), data[:,i], 0) for i in range(data.shape[1])]
            data_sws = np.array(data_sws).T
            mean_fr_sws = np.zeros(data.shape[1]) * np.nan
            mean_amp_sws = np.zeros(data.shape[1]) * np.nan

        if not(np.isnan(np.where(hypno == -3 , 1, np.nan)).all()):
            data_rem = [np.where(np.where(hypno == -3 , 1, 0), data[:,i], 0) for i in range(data.shape[1])]
            data_rem = np.array(data_rem).T
            mean_fr_rem = np.sum(np.where(data_rem > 0 , 1, 0), axis = 0) / ((np.sum(np.where(hypno == -3 ,
                                 1, 0)) - artifacts_length_rem) / sr)
            mean_amp_rem = np.sum(np.where(data_rem > 0, data_rem, 0), axis = 0) / np.sum(np.where(data_rem > 0, 
                                                        1, 0), axis = 0)
        else:
            data_rem = [np.where(np.where(hypno == -3 , 1, 0), data[:,i], 0) for i in range(data.shape[1])]
            data_rem = np.array(data_rem).T
            mean_fr_rem = np.zeros(data.shape[1]) * np.nan
            mean_amp_rem = np.zeros(data.shape[1]) * np.nan
        
        
        if not(np.isnan(np.where(hypno == 0 , 1, np.nan)).all()):

            data_awake = [np.where(np.where(hypno == 0 , 1, 0), data[:,i], 0) for i in range(data.shape[1])]
            data_awake = np.array(data_awake).T
            mean_fr_awake = np.sum(np.where(data_awake > 0 , 1, 0), 
                                    axis = 0) / ((np.sum(np.where(hypno == 0 , 1, 0)) - artifacts_length_awake) / sr)
            mean_amp_awake = np.sum(np.where(data_awake > 0, data_awake, 0), 
                                    axis = 0) / np.sum(np.where(data_awake > 0, 1, 0), axis = 0)
        else:
            data_awake = [np.where(np.where(hypno == 0 , 1, 0), data[:,i], 0) for i in range(data.shape[1])]
            data_awake = np.array(data_awake).T
            mean_fr_awake = np.zeros(data.shape[1]) * np.nan
            mean_amp_awake = np.zeros(data.shape[1]) * np.nan
        

        if not(fr_zeros):
            mean_fr_sws = np.where(mean_fr_sws!=0, mean_fr_sws, np.nan)
            mean_fr_rem = np.where(mean_fr_rem!=0, mean_fr_rem, np.nan)
            mean_fr_awake = np.where(mean_fr_awake!=0, mean_fr_awake, np.nan)

        self.data_sws = data_sws
        self.data_rem = data_rem
        self.data_awake = data_awake
        self.mean_fr_sws = mean_fr_sws
        self.mean_fr_rem = mean_fr_rem
        self.mean_fr_awake = mean_fr_awake
        self.mean_amp_sws = mean_amp_sws
        self.mean_amp_rem = mean_amp_rem
        self.mean_amp_awake = mean_amp_awake

    def get_group_data(self):

        print('output = sws, rem , awake')
        print(f'SWS size {self.data_sws.shape}, REM size {self.data_rem.shape}, awake size {self.data_awake.shape}')
        return self.data_sws, self.data_rem, self.data_awake

    def get_mean_firing_rates (self):
        print('output = sws, rem , awake')
        return self.mean_fr_sws, self.mean_fr_rem, self.mean_fr_awake
        

    def get_mean_amplitudes(self):
        print('output = sws, rem , awake')
        return self.mean_amp_sws, self.mean_amp_rem, self.mean_amp_awake


def run_per_mouse(sf_, ending_, unit_lenght_, address_, set_artifacts_, type_, fr_zeros):
    # running for all sequences in one go
    # initializing ETL
    ETL_ = ETL(sf=sf_, ending=ending_, unit_length=unit_lenght_)

    # giving file or files path
    ETL_.get_path(address_)
    
    # loading all files to ETL memory
    ETL_.load_files()

    # getting hypno signals for all files
    Hypno_ = ETL_.get_hypno()

    # getting full data set
    full_dataset = ETL_.get_data_values()

    print(f'Hypno shape {Hypno_.shape} and Data shape {full_dataset.shape}')

    if set_artifacts_:
    # setting artifact period
        for _, qq_data in enumerate(set_artifacts_):
            ETL_.set_artifact(start_= np.array(qq_data[0]), end_= np.array(qq_data[1]), file_= np.array(qq_data[2]))
            print(f'setting artifacts for file {np.array(qq_data[2])}')


    # getting artifacts
    data_artifacts, hypno_artifacts = ETL_.get_artifacts()
    print(f'data artifact shape {data_artifacts.shape}, hypno_artifact shape {hypno_artifacts.shape}')


    # initializing activation class
    if not(set_artifacts_):
        activation_ = activation(hypno=Hypno_, data=full_dataset, hypno_artifacts=False, data_artifacts=False)
    else:
        activation_ = activation(hypno=Hypno_, data=full_dataset, hypno_artifacts=hypno_artifacts, data_artifacts=data_artifacts)

    frequency_ = []
    amplitude_ = []

    for i in range(Hypno_.shape[1]):
        
        # executing mean firing rate calculation
        activation_.mean_fire(file_nr=i, sr=sf_, fr_zeros = fr_zeros)


        # getting mean firing rate result for all cells in given file
        
        frequency_.append(activation_.get_mean_firing_rates())

        # getting mean amplitude for all cells in given file
        amplitude_.append(activation_.get_mean_amplitudes())

    sws_fr = np.stack(frequency_)[:,0,:]
    rem_fr = np.stack(frequency_)[:,1,:]
    awake_fr = np.stack(frequency_)[:,2,:]

    sws_amp = np.stack(amplitude_)[:,0,:]
    rem_amp = np.stack(amplitude_)[:,1,:]
    awake_amp = np.stack(amplitude_)[:,2,:]

    print(f'shapes: sws_fr {sws_fr.shape}, rem_fr {rem_fr.shape}, sws_amp {sws_amp.shape}, rem_amp {rem_amp.shape}')



    # taking average of all recordings per mouse
    if type_=='mean' or type_=='Mean':
        sws_fr_avg = np.nanmean(sws_fr, axis=0)
        rem_fr_avg = np.nanmean(rem_fr, axis=0)
        awake_fr_avg = np.nanmean(awake_fr, axis=0)

        sws_amp_avg = np.nanmean(sws_amp, axis=0)
        rem_amp_avg = np.nanmean(rem_amp, axis=0)
        awake_amp_avg = np.nanmean(awake_amp, axis=0)

    elif type_=='median' or type_=='Median':

        sws_fr_avg = np.nanmedian(sws_fr, axis=0)
        rem_fr_avg = np.nanmedian(rem_fr, axis=0)
        awake_fr_avg = np.nanmedian(awake_fr, axis=0)

        sws_amp_avg = np.nanmedian(sws_amp, axis=0)
        rem_amp_avg = np.nanmedian(rem_amp, axis=0)
        awake_amp_avg = np.nanmedian(awake_amp, axis=0)

    print(f'shapes: sws_fr_avg {sws_fr_avg.shape}, rem_fr_avg {rem_fr_avg.shape}, awake_fr_avg {awake_fr_avg.shape}, sws_amp_avg {sws_amp_avg.shape}, rem_amp_avg {rem_amp_avg.shape}, awake_amp_avg {awake_amp_avg.shape}')

    return frequency_, amplitude_, sws_fr_avg, rem_fr_avg, awake_fr_avg, sws_amp_avg, rem_amp_avg, awake_amp_avg


def run_per_mouse_concatinated(sf_, ending_, unit_lenght_, address_, set_artifacts_, fr_zeros):

# initializing ETL
    ETL_ = ETL(sf=sf_, ending=ending_, unit_length=unit_lenght_)

    # giving file or files path
    ETL_.get_path(address_)
    
    # loading all files to ETL memory
    ETL_.load_files()

    # getting hypno signals for all files
    Hypno_ = ETL_.get_hypno()

    # getting full data set
    full_dataset = ETL_.get_data_values()

    print(f'Hypno shape {Hypno_.shape} and Data shape {full_dataset.shape}')

    if set_artifacts_:
    # setting artifact period
        for _, qq_data in enumerate(set_artifacts_):
            ETL_.set_artifact(start_= np.array(qq_data[0]), end_= np.array(qq_data[1]), file_= np.array(qq_data[2]))
            print(f'setting artifacts for file {np.array(qq_data[2])}')


    # getting artifacts
    data_artifacts, hypno_artifacts = ETL_.get_artifacts()
    print(f'data artifact shape {data_artifacts.shape}, hypno_artifact shape {hypno_artifacts.shape}')


    # initializing activation class
    if not(set_artifacts_):
        activation_ = activation(hypno=np.hstack(Hypno_.T)[:,np.newaxis], data=np.hstack(full_dataset.T).T[:,:,np.newaxis], 
                         hypno_artifacts=False, data_artifacts=False)
    else:
        activation_ = activation(hypno=np.hstack(Hypno_.T)[:,np.newaxis], data=np.hstack(full_dataset.T).T[:,:,np.newaxis], 
                             hypno_artifacts=np.hstack(hypno_artifacts)[:,np.newaxis], 
                             data_artifacts=np.hstack(data_artifacts.T).T[:,:,np.newaxis])

    activation_.mean_fire(file_nr=0, sr=sf_, fr_zeros = fr_zeros)
    sws_fr, rem_fr, awake_fr = activation_.get_mean_firing_rates()
    sws_amp, rem_amp, awake_amp = activation_.get_mean_amplitudes()
    #sws, rem, awake = activation_.get_group_data()

    return sws_fr, rem_fr, awake_fr, sws_amp, rem_amp, awake_amp #, sws, rem, awake



class Plot:

    def __init__(self, x, y, z, xlabel, ylabel, zlabel, save_address):
        self.x = x # x data as a vector
        self.y = y # y data as a vector
        self.xlabel = xlabel # name of x axes
        self.ylabel = ylabel # name of y axes
        self.z = z
        self.zlabel = zlabel
        self.save_address = save_address

    def scatter_3D(self, image_size, view_):

        fig = plt.figure(figsize=image_size) #create a canvas, tell matplotlib it's 3d
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.x, self.y, self.z, marker='.')
        
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_zlabel(self.zlabel)
        ax.set_title('3D scatter plot')

        ax.view_init(view_[0], view_[1])

        if self.save_address:
            plt.savefig(self.save_address + '/3D scatter plot.pdf', dpi=300, format='pdf')

    def hist_2d(self, bins_, range_, interpolation_):

        x = self.x
        y = self.y
        save_address = self.save_address

        range_ = np.array(range_)
        plt.hexbin(x, y, bins = bins_, cmap='Blues')   
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title('2D histogram plot')
        plt.xlim(range_[0,:])
        plt.ylim(range_[1,:])
        if interpolation_:
            from scipy.stats import gaussian_kde
            data = np.vstack([x, y])
            kde = gaussian_kde(data)

            # evaluate on a regular grid
            xgrid = np.linspace(range_[0,0], range_[0,1], bins_)
            ygrid = np.linspace(range_[1,0], range_[1,1], bins_)
            Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
            Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))

            # Plot the result as an image
            plt.imshow(Z.reshape(Xgrid.shape),
                        origin='lower', aspect='auto',
                        extent=[range_[0,0], range_[0,1], range_[1,0], range_[1,1]],
                        cmap='Blues')

            plt.xlabel(self.xlabel)
            plt.ylabel(self.ylabel)
            plt.title('2D histogram plot')
        if save_address:
            plt.savefig(save_address + '/2D Histogram.pdf', dpi=300, format='pdf')

    def hist_3d(self, bins_, range_, density_):
        
        from mpl_toolkits.mplot3d import Axes3D
        x = self.x
        y = self.y
        save_address = self.save_address

        XY = np.stack((x,y),axis=-1)

        def selection(XY, limitXY=range_):
                XY_select = []
                for elt in XY:
                    if elt[0] > limitXY[0][0] and elt[0] < limitXY[0][1] and elt[1] > limitXY[1][0] and elt[1] < limitXY[1][1]:
                        XY_select.append(elt)

                return np.array(XY_select)

        #XY_select = selection(XY, limitXY=range_)


        #xAmplitudes = np.array(XY_select)[:,0]#your data here
        #yAmplitudes = np.array(XY_select)[:,1]#your other data here


        fig = plt.figure(figsize=(10,10)) #create a canvas, tell matplotlib it's 3d
        ax = fig.add_subplot(111, projection='3d')


        hist, xedges, yedges = np.histogram2d(x, y, bins=bins_, range = range_, density = density_) # you can change your bins, and the range on which to take data
        
        xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:]) -(xedges[1]-xedges[0])


        xpos = xpos.flatten()*1./2
        ypos = ypos.flatten()*1./2
        zpos = np.zeros_like (xpos)

        dx = xedges [1] - xedges [0]
        dy = yedges [1] - yedges [0]
        dz = hist.flatten()

        cmap = plt.cm.get_cmap('Blues') # Get desired colormap - you can change this!
        max_height = np.max(dz)   # get range of colorbars so we can normalize
        min_height = np.min(dz)
        # scale each z to [0,1], and get their rgb values
        rgba = [cmap((k-min_height)/max_height) for k in dz] 

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba)#, zsort='average')
        
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        if density_:
            ax.set_zlabel('pdf')
        else:
            ax.set_zlabel('count')
        plt.show()
        ax.set_title('3D histogram plot')
        if density_:
            print('bin_count / sample_count / bin_area')
        if save_address:
            fig.savefig(save_address + '/3D Histogram.pdf', dpi=300, format='pdf')

    def surface_3d(self, bins_, range_, density_):
        from mpl_toolkits.mplot3d import Axes3D
        x = self.x
        y = self.y
        save_address = self.save_address

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        hist, xedges, yedges = np.histogram2d(x, y, bins=bins_, range=range_, density=density_)
        X, Y = np.meshgrid((xedges[1:] + xedges[:-1]) / 2,
                   (yedges[1:] + yedges[:-1]) / 2)

        # make the plot, using a "jet" colormap for colors
        ax.plot_surface(X, Y, hist, cmap='Blues')
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        if density_:
            ax.set_zlabel('pdf')
        else:
            ax.set_zlabel('count')
        ax.set_title('3D surface plot')
        plt.show()  # or plt.savefig('contours.pdf')
        if save_address:
            fig.savefig(save_address + '/3D Surface.pdf', dpi=300, format='pdf')

    def marginal_dist(self, awake_, bins_, range_, density_, kde_, name_):
        import seaborn as sns

        x = self.x
        y = self.y
        z = self.z
        save_address = self.save_address
        plt.figure(figsize=(10,10))
        
        range_ = np.array(range_)
        if density_ and kde_:
            plt.figure(figsize=(10,10))
            sns.distplot(x, bins=bins_, kde=True,  norm_hist=True, label=self.xlabel, 
            hist_kws={"histtype": 'bar', "linewidth": 3, 'rwidth' :0.8})
            sns.distplot(y, bins=bins_, kde=True,  norm_hist=True, label=self.ylabel, 
            hist_kws={"histtype": 'bar', "linewidth": 3, 'rwidth' :0.8})
            if awake_:
                sns.distplot(z, bins=bins_, kde=True,  norm_hist=True, label=self.zlabel, 
                hist_kws={"histtype": 'bar', "linewidth": 3, 'rwidth' :0.8})
            plt.legend(prop={'size': 10}, title = 'Hypno groups')
            plt.title('Density Plot with Multiple Hypno groups')
            plt.xlabel(name_)
            plt.ylabel('Density')
            plt.xlim(range_[0,:])
            plt.ylim(range_[1,:])

        elif density_ and not(kde_):
            plt.figure(figsize=(10,10))
            sns.distplot(x, bins=bins_, kde=False,  norm_hist=True, label=self.xlabel, 
            hist_kws={"histtype": 'bar', "linewidth": 3, 'rwidth' :0.8})
            sns.distplot(y, bins=bins_, kde=False,  norm_hist=True, label=self.ylabel, 
            hist_kws={"histtype": 'bar', "linewidth": 3, 'rwidth' :0.8})
            if awake_:
                sns.distplot(z, bins=bins_, kde=False,  norm_hist=True, label=self.zlabel, 
                hist_kws={"histtype": 'bar', "linewidth": 3, 'rwidth' :0.8})
            plt.legend(prop={'size': 10}, title = 'Hypno groups')
            plt.title('Density Plot with Multiple Hypno groups')
            plt.xlabel(name_)
            plt.ylabel('Density')
            plt.xlim(range_[0,:])
            plt.ylim(range_[1,:])
        
        elif not(density_) and not(kde_):
            plt.figure(figsize=(10,7))
            sns.distplot(x, bins=bins_, kde=False,  norm_hist=False, label=self.xlabel, 
            hist_kws={"histtype": 'bar', "linewidth": 3, 'rwidth' :0.8})
            sns.distplot(y, bins=bins_, kde=False,  norm_hist=False, label=self.ylabel, 
            hist_kws={"histtype": 'bar', "linewidth": 3, 'rwidth' :0.8})
            if awake_:
                sns.distplot(z, bins=bins_, kde=False,  norm_hist=False, label=self.zlabel, 
                hist_kws={"histtype": 'bar', "linewidth": 3, 'rwidth' :0.8})
            plt.legend(prop={'size': 10}, title = 'Hypno groups')
            plt.title('Density Plot with Multiple Hypno groups')
            plt.xlabel(name_)
            plt.ylabel('Counts')
            plt.xlim(range_[0,:])
            plt.ylim(range_[1,:])


        if save_address:
            plt.savefig(save_address + '/marginal distributions.pdf', dpi=300, format='pdf')

    def variation_plot(self, data_, figure_size, x_label_, range_, jitter_):

        range_ = np.array(range_)
        save_address = self.save_address
        df = pd.DataFrame(columns=['SWS', 'REM', 'awake', 'labels'])

        for i, res in enumerate(data_):
            
            df = df.append(pd.DataFrame({'SWS' : res[0], 'REM': res[1], 'awake': res[2], 
            'labels': np.repeat('sequence' + str(i+1),len(res[0]))}), ignore_index = True)

        df_melt = df.melt(id_vars= 'labels', var_name='Hypno_groups')
        

        # Initialize the figure
        f, ax = plt.subplots(figsize=figure_size)
        
        # Show each observation with a scatterplot
        sns.stripplot(x="value", y="Hypno_groups", hue="labels",
              data=df_melt, dodge=True, alpha=.25, zorder=1, jitter=jitter_)

        ax.set_xlabel(x_label_)
        ax.set_title('distribution per run for hypno groups')
        ax.set_xlim(range_)
        if save_address:
            f.savefig(save_address + '/Strip plot.pdf', dpi=300, format='pdf')


    def join_plot(self, bins_, kind_, robust_):

        x = self.x
        y = self.y
        save_address = self.save_address

        plt.figure(figsize = (10,10))
        g = sns.jointplot(x=x, y=y, kind=kind_, color="#4CB391", joint_kws={'robust':robust_},
        marginal_kws=dict(bins=bins_, fill=True))
        g.set_axis_labels(xlabel=self.xlabel, ylabel=self.ylabel)
        range__ = [bins_[0], bins_[-1]]
        plt.xlim([bins_[0], bins_[-1]])
        plt.ylim([bins_[0], bins_[-1]])
        plt.plot(range__, range__, ls = '--', c = 'k')
        if save_address:
            g.savefig(save_address + '/Joint plot.pdf', dpi=300, format='pdf')

    def join_plotV2(self, bins_, kind_, robust_, xlim_, ylim_):

        x = self.x
        y = self.y
        save_address = self.save_address

        plt.figure(figsize = (10,10))
        g = sns.jointplot(x=x, y=y, kind=kind_, color="#4CB391", joint_kws={'robust':robust_},
        marginal_kws=dict(bins=bins_, fill=True), xlim = xlim_, ylim = ylim_)
        g.set_axis_labels(xlabel=self.xlabel, ylabel=self.ylabel)
        if save_address:
            g.savefig(save_address + '/Joint plot.pdf', dpi=300, format='pdf')
        

    def state_dependent_plot(self, based_on, pr_to_take, kde_, bins_, stat_, fill_, figsize_, kind_, least_, robust_):

        """
        based_on is a parameter indicated dataframe should be sotd based on which of x, y, z
                    please give labels which is used for xlabel or ylabel or ...
        
        pr_to_take is percentageof samples we want to take; 5, 10, 20 , ...

        stat : {"count", "frequency", "density", "probability"}
        least_ : True or False when True sorts small to big when False sorts big to small
        """
        # creating data frame
        df = pd.DataFrame({self.xlabel:self.x, self.ylabel:self.y, self.zlabel:self.z})
        
        # sorting data based on column
        df_sorted = df.sort_values(by=based_on, ascending=least_)

        # selecting portion of data
        df_selected = df_sorted[0:np.ceil((pr_to_take/100) * df.shape[0]).astype(np.int32)]
        
        #start plotting
        g0 = sns.histplot(data = df_selected, stat = stat_, fill=fill_, 
                            kde=kde_, bins = bins_)
        if self.save_address:
            g0.figure.savefig(self.save_address + '/state_dependent_histogram.pdf', dpi=300, format='pdf')

        
        plt.figure(figsize = (10,10))
        g1 = sns.jointplot(data = df_selected, x=self.xlabel, y=self.zlabel, kind=kind_, color="#4CB391",
        marginal_kws=dict(bins=bins_, fill=True), joint_kws={'robust':robust_})
        g1.set_axis_labels(xlabel=self.xlabel, ylabel=self.zlabel)
        range__ = [bins_[0], bins_[-1]]
        plt.plot(range__, range__, ls = '--', c = 'k')
        plt.xlim([bins_[0], bins_[-1]])
        plt.ylim([bins_[0], bins_[-1]])
        if self.save_address:
            g1.savefig(self.save_address + '/state_dependent_jointPlot' + self.xlabel + '-' + self.zlabel + '.pdf', 
             dpi=300, format='pdf')


        plt.figure(figsize = (10,10))
        g2 = sns.jointplot(data = df_selected, x=self.xlabel, y=self.ylabel, kind=kind_, color="#4CB391",
        marginal_kws=dict(bins=bins_, fill=True), joint_kws={'robust':robust_})
        g2.set_axis_labels(xlabel=self.xlabel, ylabel=self.ylabel)
        range__ = [bins_[0], bins_[-1]]
        plt.xlim([bins_[0], bins_[-1]])
        plt.ylim([bins_[0], bins_[-1]])
        plt.plot(range__, range__, ls = '--', c = 'k')
        if self.save_address:
            g2.savefig(self.save_address + '/state_dependent_jointPlot' + self.xlabel + '-' + self.ylabel + '.pdf',
             dpi=300, format='pdf')


        plt.figure(figsize = (10,10))
        g3 = sns.jointplot(data = df_selected, x=self.zlabel, y=self.ylabel, kind=kind_, color="#4CB391",
        marginal_kws=dict(bins=bins_, fill=True), joint_kws={'robust':robust_})
        g3.set_axis_labels(xlabel=self.zlabel, ylabel=self.ylabel)
        range__ = [bins_[0], bins_[-1]]
        plt.xlim([bins_[0], bins_[-1]])
        plt.ylim([bins_[0], bins_[-1]])
        plt.plot(range__, range__, ls = '--', c = 'k')
        if self.save_address:
            g3.savefig(self.save_address + '/state_dependent_jointPlot' + self.zlabel + '-' + self.ylabel + '.pdf',
             dpi=300, format='pdf')
        

        





# This plot is outside of Plot class
def raster_plot(data_set, hypno_, sr, dot_size, type, palette_, bins_, xlim_, kde_, save_address):

    import seaborn as sns
    import matplotlib.colors as colors
    import matplotlib.cm as cmx

    #creating proper colormap for lines
    cm = plt.get_cmap('Blues') 
    cNorm  = colors.Normalize(vmin=0, vmax=6)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    #creating correct data format
    data_index = np.where(data_set>0)
    frames = data_index[0]
    cells = data_index[1]
    amplitudes = data_set[frames, cells]
    hypno = hypno_[frames]
    hypno = [str(i) for i in hypno]
    hypno = np.char.replace(hypno, '-2.0', 'sws')
    hypno = np.char.replace(hypno, '-3.0', 'rem')
    hypno = np.char.replace(hypno, '0.0', 'awake')
    df = pd.DataFrame({'frames':frames, 'cells': cells, 
                    'times':frames/sr, 'amplitude':amplitudes, 'Groups': hypno})

    # creating background
    test = np.zeros((data_set.shape[1],len(hypno_)))
    index_sws = np.where(hypno_ == -2)
    index_rem = np.where(hypno_ == -3)
    index_awake = np.where(hypno_ == 0)

    test[:,index_sws] = 3
    test[:,index_rem] = 6
    test[:,index_awake] = 0

    Test = test[:,np.int16(np.linspace(0,len(hypno_)-1,600))]

        # start plotting


    if type==1:
        
        g = sns.relplot(
            data=df,
            x="times", y="cells",
            hue="Groups", size="amplitude",
            sizes=dot_size, palette=palette_
        )
        g.ax.set_xlim(xlim_)
        g.fig.set_figwidth(12)
        g.fig.set_figheight(6)
        plt.title('Raster Plot')
        
    elif type==2:

        g = sns.relplot(data=df,
            x="times", y="cells",
            size="amplitude",
            sizes=(2, 60)) 
        g.ax.set_xlim(xlim_)
        g.fig.set_figwidth(12)
        g.fig.set_figheight(6)
        line1 = g.ax.scatter(0,1, s = 0.01, c = scalarMap.to_rgba(1), alpha = 0.4)
        line2 = g.ax.scatter(0,1, s = 0.01, c = scalarMap.to_rgba(3), alpha = 0.4)
        line3 = g.ax.scatter(0,1, s = 0.01, c = scalarMap.to_rgba(6), alpha = 0.4)

        g.ax.imshow(Test, alpha=.4, aspect='auto', cmap='Blues', vmin=0, vmax = 6 )

        g.ax.legend((line1, line2, line3),('awake','sws','rem'), loc='center', 
                    bbox_to_anchor=(0.75, -0.05, 0.92, 0.48), title = 'Hypno', borderaxespad = None, 
                edgecolor = None, facecolor=None, framealpha = 1.0, frameon = False, 
                markerscale = 100.0)
        plt.title('Raster Plot')
        

        
    elif type==3:

        g = sns.relplot(data=df,
            x="times", y="cells",
            hue="amplitude", palette=palette_) 
        g.ax.set_xlim(xlim_)
        g.fig.set_figwidth(12)
        g.fig.set_figheight(6)

        line1 = g.ax.scatter(0,1, s = 0.01, c = scalarMap.to_rgba(1), alpha = 0.4)
        line2 = g.ax.scatter(0,1, s = 0.01, c = scalarMap.to_rgba(3), alpha = 0.4)
        line3 = g.ax.scatter(0,1, s = 0.01, c = scalarMap.to_rgba(6), alpha = 0.4)

        g.ax.imshow(Test, alpha=.4, aspect='auto', cmap='Blues', vmin=0, vmax = 6 )

        g.ax.legend((line1, line2, line3),('awake','sws','rem'), loc='center', 
                    bbox_to_anchor=(0.75, -0.05, 0.92, 0.48), title = 'Hypno', borderaxespad = None, 
                edgecolor = None, facecolor=None, framealpha = 1.0, frameon = False, 
                markerscale = 100.0)
        plt.title('Raster Plot')
        
    if type==1:
        ax2 = g.fig.add_axes([.1,1.05,.73,0.2])
    elif type==2 or type==3:
        ax2 = g.fig.add_axes([.1,1.05,.75,0.2])
    sns.distplot(df['times'], kde = kde_, bins = bins_, ax = ax2)
    ax2.set_xlim(xlim_)
    ax2.set_axis_off()
    if save_address:
        g.savefig(save_address + '/raster plot.pdf', dpi=300, format='pdf')
        



class epoch_duration:
    """
    this calss calculating epoch duration for given unit
    hypnos is numpy array contains hypno signal 

    """
    def __init__(self, hypnos, data, data_artifact, hypno_artifact, sf, save_address):

        # initializing calss
        self.hypnos = hypnos
        self.data = data
        self.data_artifact = data_artifact
        self.hypno_artifact = hypno_artifact
        self.sf = sf
        self.save_address = save_address
        self.prepare_hypno()
        self.add_avg_firing_rate()

    def prepare_hypno(self):

        """
        state_name --> 'SWS', 'REM', 'Awake'

        """

        df = findseq(np.squeeze(self.hypnos))
        df['duration_sec'] = df.duration / self.sf
        df['state_names'] = np.nan
        df.state_names[df.state == -2] = 'SWS'
        df.state_names[df.state == -3] = 'REM'
        df.state_names[df.state == 0] = 'Awake'

        self.hypno_df = df

    def add_avg_firing_rate(self):
        
        # loading seqs dataframe
        df = self.hypno_df

        # preparing data
        data = self.data * np.logical_not(self.data_artifact)
        data = np.where(data>0,1,0)

        df['avg_firing_rate_with_zero'] = np.nan
        df['avg_firing_rate_without_zero'] = np.nan

        # calculating firing rate per second for all cells and then take average of it
        for i in range(df.shape[0]):
            # how we calculate average firing rate per epoch
            # do we need to keep zeros when averaging or take them out
            ind1 = df.start_index[i]
            ind2 = df.end_index[i]

            avg_ = np.nansum(data[ind1:ind2,:], axis=0) / (df.duration[i] / self.sf) # per second

            df.avg_firing_rate_with_zero[i] = np.nanmean(avg_)
            df.avg_firing_rate_without_zero[i] = np.nanmean(avg_[avg_>0])

        self.hypno_df = df

    def load_df(self, df):
        self.hypno_df = df

    def get_df(self):

        return self.hypno_df

    def load_pic_save_add(self, address):
        self.save_address = address

    def plot_duration_correlation(self, with_zero, xlim_, ylim_):

        df = self.hypno_df
        
        if with_zero:
            g = sns.FacetGrid(df, col="state_names", margin_titles=True)
            g.map_dataframe(sns.scatterplot, x="duration_sec", y= "avg_firing_rate_with_zero")

        else:
            g = sns.FacetGrid(df, col="state_names", margin_titles=True)
            g.map_dataframe(sns.scatterplot, x="duration_sec", y= "avg_firing_rate_without_zero")
        
        g.tight_layout()
        g.set_axis_labels("Duration (sec)", "Average firing rate")
        g.set(xlim=xlim_, ylim = ylim_)

        if self.save_address:
            g.savefig(self.save_address + '/Duration correlation plot.pdf', dpi=300, format='pdf')






    def plot_duration(self, figsize_, state_name, kde_, range_):

        """
        state_name --> 'SWS', 'REM', 'Awake'

        """

        df = self.hypno_df
        
        tmp_d = df.duration_sec[df.state_names==state_name].values
        f, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize_)
        ax.hist(tmp_d, density = kde_, bins = range_, edgecolor='k', alpha=0.75)
        ax.set_xlabel('epoch duration (sec)')
        if kde_:
            ax.set_ylabel('Density')
        else:
            ax.set_ylabel('Count')
        ax.set_title(state_name)
        if self.save_address:
            f.savefig(self.save_address + '/Duration plot.pdf', dpi=300, format='pdf')
        

        
def findseq(x):

    # this is a function to calculate sequences and create table
    # table header start_value, start_index, end_index, duration

    t0 = x[:-1]
    t1 = x[1:]
    diff_ = t1 - t0
    # finding change points
    changes = np.where(diff_!=0)  # this is the end points in each epoch and after this change happens
    changes = np.array(changes[0]).astype(np.int32)
    seq = np.zeros((len(changes) + 1, 4)).astype(np.int32)
    
    for i, change in enumerate(changes):
        
        if i == 0:
            seq[i,0] = t0[i]
            seq[i,1] = i
            seq[i,2] = change
            seq[i,3] = change - i
        else:
            seq[i,0] = t0[change]
            seq[i,1] = changes[i-1] + 1
            seq[i,2] = change
            seq[i,3] = change - (changes[i-1])
            
    seq[-1,0] = t0[changes[-1] + 1]
    seq[-1,1] = changes[-1] + 1
    seq[-1,2] = len(t0) +1
    seq[-1,3] = (len(t0) +1) - (changes[-1])

    df = pd.DataFrame({'state': seq[:,0],'start_index': seq[:,1],'end_index': seq[:,2],'duration': seq[:,3]})
    return df




class transition:
    """
    we are looking to transition graph from one state to another one

    """
    def __init__(self, data, hypno, data_artifact, hypno_artifact, sf, pattern, window_size):

        """
        pattern has to be in dictionary form {'SWS':-2, 'REM':-3}

        window_size [4, 10]
        """

        if len(pattern) != 2:
            raise Exception("Length of transition pattern should be 2")

        if len(hypno) != data.shape[0]:
            raise Exception('data length is not equal to hypno length or you need to tanspose data')

        self.data = data
        self.hypno = hypno
        self.data_artifact = data_artifact
        self.hypno_artifact = hypno_artifact
        self.sf = sf
        self.pattern = pattern
        self.seqs = self.find_seq()
        self.window_size = window_size

        self.window_check_result = self.check_window()
        self.artifact_index = self.check_artifacts()

    def find_seq(self):

        seqs = findseq(self.hypno)
        return seqs.to_numpy()

    def check_window(self):

        # loading data
        seqs = self.seqs
        pattern = self.pattern
        window = self.window_size
        # get keys of patterns
        keys = list(pattern)
        
        # check duration of seqs for given pattern and compare it corresponding window size
        patt1_index = np.where((seqs[:,0] == pattern[keys[0]]) & (seqs[:,3] < window[0] * self.sf))[0]
        patt2_index = np.where((seqs[:,0] == pattern[keys[1]]) & (seqs[:,3] < window[1] * self.sf))[0]

        # all patterns regardless of windows
        patt1_all = np.where(seqs[:,0] == pattern[keys[0]] )[0]
        patt2_all = np.where(seqs[:,0] == pattern[keys[1]] )[0]

        warnings.warn('comparing epochs duration and given window length')
        print(f'From all {patt1_all.shape[0]} as first section of search pattern, duration of {patt1_index.shape[0]} of them are smaller than window length and will be removed')
        print(f'From all {patt2_all.shape[0]} as second section of search pattern, duration of {patt2_index.shape[0]} of them are smaller than window length and will be removed')

        return [patt1_index, patt2_index]


    def check_artifacts(self):
        
        # here I am cheking if artifacts exist during selected patterns
        # If exist I then remove that combination

        # load artifacts
        
        hypno_art = self.hypno_artifact
        seqs = self.seqs
        window = self.window_size

        art_index = np.zeros(seqs.shape[0])
        for i in range(seqs.shape[0]):

            artifact_ = hypno_art[seqs[i,2]-window[0] :seqs[i,2]+window[1]]
            if artifact_.sum()>0:
                art_index[i] = 1
            # artifact index is based on location in first pattern


        return art_index


    def get_transition(self, bin_size, norm_):
        """
        bin_size in sec
        norm_ is normalize histogram or not 
        """

        # loading data
        seqs = self.seqs
        pattern = self.pattern
        window = self.window_size
        window_check_result = self.window_check_result
        artifact_index = self.artifact_index

        # get keys of patterns
        keys = list(pattern)
        search_pattern = np.array([pattern[keys[0]], pattern[keys[1]]])

        all_stats = np.array(seqs[:,0])

        # search for location of patterns
        locs = np.where((all_stats[:-1] == search_pattern[0]) & (all_stats[1:] == search_pattern[1]))[0]
        locs = np.int32(locs)
        if locs.shape[0] == 0:
            raise Exception('there is no such a pattern in Hypno')
            
       
        counts = []
        for _, loc in enumerate(locs):
            
            if not((loc == window_check_result[0]).any()) and not(((loc+1) == window_check_result[1]).any()) and (artifact_index[loc]!=1) and (artifact_index[loc+1]!=1): #check all filters(window_check_result, artifact_index)
                
                # taking epoch of data given found pattern
                signal = []; frames = []; indexes = [] #cells = [];
                signal = self.data[seqs[loc,2]-(np.int32(window[0] * self.sf)):seqs[loc,2]+(np.int32(window[1] * self.sf)),:]

                # finding indices of activities
                indexes = np.where(signal>0)
                frames = indexes[0]
                #cells = indexes[1]

                # calculate time from frames and shift by window[0]
                hist_ = np.histogram((frames / self.sf) - window[0], 
                                    bins = np.arange(-window[0],window[1]+bin_size, bin_size))
                # normalization comes to normalize numbers per bin and keep reporting x per second
                if norm_:
                    counts.append(hist_[0] / bin_size)
                else:
                    counts.append(hist_[0])
                hist_locs = hist_[1]

        counts = np.stack(counts)

        self.hist_counts = counts
        self.hist_locs = hist_locs
        self.bin_size = bin_size
        self.hist_norm = norm_

    
    def get_hist_values(self):

        return self.hist_counts, self.hist_locs

    def load_hist_values(self, hist_counts, hist_locs):

        self.hist_counts = hist_counts
        self.hist_locs = hist_locs


    def plot_transition(self, save_address):
        
        counts = self.hist_counts
        bins_locs = self.hist_locs
        bins_size = self.bin_size

        bins_locs = bins_locs[:-1] + bins_size /2
        # calculate mean and std
        mean_ = np.nanmean(counts, axis = 0)
        #std_ = np.nanstd(counts, axis = 0)
        std_ = np.nanstd(counts, axis = 0) / np.sqrt(counts.shape[0])  

        f, ax = plt.subplots(1,1,figsize=(8,5))
        ax.plot(bins_locs, mean_, linewidth=2, color = 'blue')
        ax.fill_between(bins_locs, mean_ - std_, mean_ + std_, interpolate=True, alpha = 0.3)
        ax.axvline(x=0 , ls = '--', color = 'k')

        if self.hist_norm:
            ax.set_ylabel('normalized number of active cells per second')

        else:
            ax.set_ylabel('number of active cells')

        ax.set_title(f"Transition from {list(self.pattern)[0]} to {list(self.pattern)[1]}")
        ax.set_xlabel('Time (sec)')

        ax.legend(['mean', 'change_point', 'S.E.M'])

        if save_address:
            f.savefig(save_address + '/Transition plot.pdf', dpi=300, format='pdf')
        
    def get_info(self):
        print('Transition analysis from one state to another following state given hypno file.')
        print('In this analysis after selecting epoch of states based on given pattern, each epoch is')
        print('controled for its length(in time) and for having artifact. Then only window length of')
        print('state is selected and number of active cell is calculating using user defined bin_size')
        print('In the case of normalized result, numbers of active cells per bin is normalized to bin length')
        print('and results are reported in number of active cells per second.')
        print('In case of concatinated data, it is highly recommended to NORMALIZE results.')




        

    

class Triplet:

    def __init__(self, data, hypno, data_artifact, hypno_artifact, sf, pattern, normalize, min_window = 0, max_window = np.inf):


        # if pattern is empty or not
        if len(pattern['names']) == 0:
            raise Exception('no pattern is given')

        if pattern['divid'] == 0 or type(pattern['divid']) !=int:
            raise Exception('divid must be integer greater than 0')

        if len(pattern['names']) >3:
            raise Exception('you can not select more than 3 pattern. Software is limitted to max 3 patterns.')


        # check if patter is in dectionary format
        if list(pattern.keys()) != ['names', 'code', 'sequence', 'divid']:
            raise Exception('given pattern is not correct dictionary. It has to be in the form of {names:[x], code:[x], sequence:[x], divid:[x]}')

        # checking type of inputs
        for key in pattern.keys():
            if key != 'divid' and type(pattern[key]) != list:
                raise Exception(f'type of input in {key} is NOT list. All inputs must be in list format.')

        # after checking keys now I check each key
        pattern_name_length = len(pattern['names'])
        pattern_code_length = len(pattern['code'])
        pattern_sequence_length = len(pattern['sequence'])

        if (pattern_name_length != pattern_code_length):
            raise Exception('length of code is NOT equal to length of names')
        
        if (pattern_name_length-1 != pattern_sequence_length):
            raise Exception('length of names/code is NOT matching to length of sequence. len(sequence) must be len(names/code) -1.')

        if (pattern_sequence_length == 2) and (pattern['sequence'][0] != 1): # first 2 pattern need to be locked together (sequence)
            raise Exception('first 2 elements of pattern need to be locked together')

        if len(hypno) != data.shape[0]:
            raise Exception('data length is not equal to hypno length or you need to tanspose data')


        ############################ Finished controling  inputs ########################################
        # extracting pattern type
        if pattern_name_length == 1:
            pattern_type = 1
        elif pattern_name_length == 2:
            pattern_type = 2
        elif pattern_name_length == 3:
            pattern_type = 3


        # internally storing the initial data
        self.data = data
        self.hypno = hypno
        self.data_artifact = data_artifact
        self.hypno_artifact = hypno_artifact
        self.sf = sf
        self.pattern = pattern
        self.seqs = self.find_seq()
        self.pattern_type = pattern_type
        self.normalize = normalize
        self.min_window = min_window * sf # in sample point length
        self.max_window = max_window * sf

        self.artifact_index = self.check_artifacts()


        #output space
        self.type1_amp = []
        self.type1_fr = []

    def find_seq(self):

        seqs = findseq(self.hypno)
        return seqs.to_numpy()

    def check_artifacts(self):
        
        # here I am cheking if artifacts exist during selected patterns
        # If exist I then remove that combination

        # load artifacts
        
        hypno_art = self.hypno_artifact
        seqs = self.seqs

        art_index = np.zeros(seqs.shape[0])
        for i in range(seqs.shape[0]):

            artifact_ = hypno_art[seqs[i,1] :seqs[i,2]]
            if artifact_.sum()>0:
                art_index[i] = 1
            # artifact index is based on location in first pattern
        
        return art_index


    def type1(self):
        # here I look to type pattern only
        pattern_code = self.pattern['code']
        # check again
        if len(pattern_code)>1:
            raise Exception('Pattern is not type 1')
        
        # loading
        seqs = self.seqs
        data = self.data
        sf = self.sf

        # remove artifacts
        seqs = np.delete(seqs, np.where(self.artifact_index>0)[0], axis = 0)

        # apply window size filtering
        seqs = seqs[(seqs[:,3]<self.max_window) & (seqs[:,3]>self.min_window), :]

        #select specific pattern
        new_seq = seqs[seqs[:,0] == pattern_code]

        
        avg_fr = np.zeros((new_seq.shape[0], self.pattern['divid']))
        avg_amp = np.zeros((new_seq.shape[0], self.pattern['divid']))
        if self.normalize:
            # need to be code
            print('under construction :))')
        else:

            for i in range(new_seq.shape[0]):
                intervals = np.linspace(new_seq[i,1], new_seq[i,2], self.pattern['divid'] + 1, dtype= int)

                for ii in range(len(intervals) -1):
                    temp_data = []
                    temp_data = data[intervals[ii]:intervals[ii+1],:]
                    temp_loc = []
                    temp_loc = np.where(temp_data>0)
                    avg_amp[i,ii] = temp_data[temp_loc[0], temp_loc[1]].mean()
                    avg_fr[i,ii] = len(temp_loc[0]) / (temp_data.shape[0]/sf)  #normalize to length in sec

        self.type1_amp = avg_amp
        self.type1_fr = avg_fr

        return avg_amp, avg_fr


    def type2(self):

        # here I look to type 2 pattern only
        pattern_code = self.pattern['code']
        pattern_sequence = self.pattern['sequence']
        # check again
        if len(pattern_code) != 2:
            raise Exception('Pattern is not type 2')

        if len(pattern_sequence) != 1:
            raise Exception('Pattern sequence is not type 2')

        if np.array(pattern_sequence) != 1:
            raise Exception(f'Type 2 pattern search can have only code 1, but code  {pattern_sequence} is given')
        
        # loading
        seqs = self.seqs
        data = self.data
        sf = self.sf

        # remove artifacts
        seqs = np.delete(seqs, np.where(self.artifact_index>0)[0], axis = 0)

        # apply window size filtering
        seqs = seqs[(seqs[:,3]<self.max_window) & (seqs[:,3]>self.min_window), :]

        # search for location of patterns
        all_stats = np.array(seqs[:,0])

        locs = np.where((all_stats[:-1] == pattern_code[0]) & (all_stats[1:] == pattern_code[1]))[0]
        locs = np.int32(locs)
        if locs.shape[0] == 0:
            raise Exception('there is no such a pattern in Hypno')


        avg_fr = np.zeros((len(locs), 2 * self.pattern['divid']))
        avg_amp = np.zeros((len(locs), 2 * self.pattern['divid']))

        if self.normalize:
            # need to be code
            print('under construction :))')
        else:

            for i in range(len(locs)):
                intervals1 = np.linspace(seqs[locs[i],1], seqs[locs[i],2], self.pattern['divid'] + 1, dtype= int)
                intervals2 = np.linspace(seqs[locs[i] + 1,1], seqs[locs[i] + 1,2], self.pattern['divid'] + 1, dtype= int)

                for ii in range(len(intervals1) -1):

                    temp_data1 = []; temp_data2 = []
                    temp_data1 = data[intervals1[ii]:intervals1[ii+1],:]
                    temp_data2 = data[intervals2[ii]:intervals2[ii+1],:]

                    temp_loc1 = []; temp_loc2 = []
                    temp_loc1 = np.where(temp_data1>0)
                    temp_loc2 = np.where(temp_data2>0)

                    avg_amp[i,ii] = temp_data1[temp_loc1[0], temp_loc1[1]].mean()
                    avg_amp[i,ii + self.pattern['divid']] = temp_data2[temp_loc2[0], temp_loc2[1]].mean()

                    avg_fr[i,ii] = len(temp_loc1[0]) / (temp_data1.shape[0]/sf)  #normalize to length in sec
                    avg_fr[i,ii + self.pattern['divid']] = len(temp_loc2[0]) / (temp_data2.shape[0]/sf)  #normalize to length in sec


        self.type2_amp = avg_amp
        self.type2_fr = avg_fr

        return avg_amp, avg_fr



    def type3(self):

        # here I look to type 3 pattern only
        pattern_code = self.pattern['code']
        pattern_sequence = self.pattern['sequence']
        # check again
        if len(pattern_code) != 3:
            raise Exception('Pattern is not type 3')

        if len(pattern_sequence) != 2:
            raise Exception('Pattern sequence is not type 3')

        if np.array(pattern_sequence[0]) != 1:
            raise Exception(f'Type 3 pattern search can have only code 1 at the beginning, but code  {pattern_sequence[0]} is given')
        
        # loading
        seqs = self.seqs
        data = self.data
        sf = self.sf

        # remove artifacts
        seqs = np.delete(seqs, np.where(self.artifact_index>0)[0], axis = 0)

        # apply window size filtering
        seqs = seqs[(seqs[:,3]<self.max_window) & (seqs[:,3]>self.min_window), :]

        # search for location of patterns
        all_stats = np.array(seqs[:,0])

        # search pattern
        # 1. if the last state locked with previous states
        if pattern_sequence[1] == 1:
            locs = np.where((all_stats[:-2] == pattern_code[0]) & (all_stats[1:-1] == pattern_code[1]) & (all_stats[2:] == pattern_code[2]))[0]
            locs = np.int32(locs)

        # 2. if the last state not locked with previous states
        elif pattern_sequence[1] == 0:
            locs = np.where((all_stats[:-1] == pattern_code[0]) & (all_stats[1:] == pattern_code[1]))[0]
            locs_third_temp = np.where(all_stats == pattern_code[2])[0]
            # creating emty space for third pattern locations
            locs_third = []

            # going through for loop and searching third pattern
            for i in range(len(locs)):
                if i+1 < len(locs):
                    index_3 = []
                    index_3 = np.where((locs[i]+1 < locs_third_temp) & (locs[i+1] >= locs_third_temp))[0]
                    if index_3.size:
                        locs_third.append(locs_third_temp[index_3[0]])
                    else:
                        locs_third.append(np.nan)
                else:
                    index_3 = []
                    index_3 = np.where((locs[i]+1 < locs_third_temp))[0]
                    if index_3.size:
                        locs_third.append(locs_third_temp[index_3[0]])
                    else:
                        locs_third.append(np.nan)
            
            # if nan happens remove from both locs and locs_third
            locs = np.delete(locs, np.where(np.isnan(locs_third))[0]).astype(int)
            locs_third = np.delete(locs_third, np.where(np.isnan(locs_third))[0]).astype(int) 

        else:
            raise Exception(f'given pattern code is wrong. It has to start with 1 and second digit can only 0 or 1 but {pattern_sequence[1]} is given')
            
        if locs.shape[0] == 0:
            raise Exception('there is no such a pattern in Hypno')


        avg_fr = np.zeros((len(locs), 3 * self.pattern['divid']))
        avg_amp = np.zeros((len(locs), 3 * self.pattern['divid']))

        if self.normalize:
            # need to be code
            print('under construction :))')
        else:

            for i in range(len(locs)):
                if pattern_sequence[1] == 1:
                    intervals1 = np.linspace(seqs[locs[i],1], seqs[locs[i],2], self.pattern['divid'] + 1, dtype= int)
                    intervals2 = np.linspace(seqs[locs[i] + 1,1], seqs[locs[i] + 1,2], self.pattern['divid'] + 1, dtype= int)
                    intervals3 = np.linspace(seqs[locs[i] + 2,1], seqs[locs[i] + 2,2], self.pattern['divid'] + 1, dtype= int)
                elif pattern_sequence[1] == 0:
                    intervals1 = np.linspace(seqs[locs[i],1], seqs[locs[i],2], self.pattern['divid'] + 1, dtype= int)
                    intervals2 = np.linspace(seqs[locs[i] + 1,1], seqs[locs[i] + 1,2], self.pattern['divid'] + 1, dtype= int)
                    intervals3 = np.linspace(seqs[locs_third[i],1], seqs[locs_third[i],2], self.pattern['divid'] + 1, dtype= int)

                for ii in range(len(intervals1) -1):

                    temp_data1 = []; temp_data2 = []; temp_data3 = []
                    temp_data1 = data[intervals1[ii]:intervals1[ii+1],:]
                    temp_data2 = data[intervals2[ii]:intervals2[ii+1],:]
                    temp_data3 = data[intervals3[ii]:intervals3[ii+1],:]

                    temp_loc1 = []; temp_loc2 = []; temp_loc3 = []
                    temp_loc1 = np.where(temp_data1>0)
                    temp_loc2 = np.where(temp_data2>0)
                    temp_loc3 = np.where(temp_data3>0)

                    avg_amp[i,ii] = temp_data1[temp_loc1[0], temp_loc1[1]].mean()
                    avg_amp[i,ii + self.pattern['divid']] = temp_data2[temp_loc2[0], temp_loc2[1]].mean()
                    avg_amp[i,ii + 2 * self.pattern['divid']] = temp_data3[temp_loc3[0], temp_loc3[1]].mean()


                    avg_fr[i,ii] = len(temp_loc1[0]) / (temp_data1.shape[0]/sf)  #normalize to length in sec
                    avg_fr[i,ii + self.pattern['divid']] = len(temp_loc2[0]) / (temp_data2.shape[0]/sf)  #normalize to length in sec
                    avg_fr[i,ii + 2 * self.pattern['divid']] = len(temp_loc3[0]) / (temp_data3.shape[0]/sf)  #normalize to length in sec


        self.type3_amp = avg_amp
        self.type3_fr = avg_fr

        return avg_amp, avg_fr



    def type3_prior_class(self, prior_class, subset = {'Type' : 'All', 'Percentage': 0.2}):

        # here I look to type 3 pattern only
        pattern_code = self.pattern['code']
        pattern_sequence = self.pattern['sequence']
        # check again
        if len(pattern_code) != 3:
            raise Exception('Pattern is not type 3')

        if len(pattern_sequence) != 2:
            raise Exception('Pattern sequence is not type 3')

        if np.array(pattern_sequence[0]) != 1:
            raise Exception(f'Type 3 pattern search can have only code 1 at the beginning, but code  {pattern_sequence[0]} is given')

        if (np.array(prior_class['code']) == np.array(pattern_code)).any():
            raise Exception('Given prior class can not be in main pattern.')

        if len(prior_class['code']) > 1:
            raise Exception('Length of prior class can be only 1.')

        if len(prior_class['code']) < 1:
            raise Exception('No Class is provided, you have give one class')

        if (len(prior_class['sequence']) != 1):
            raise Exception('Length of sequence must be 1')

        if subset['Percentage'] > 1:
            raise Exception('Percentage must be smaller than 1')
        
        # loading
        seqs = self.seqs
        data = self.data
        sf = self.sf

        prior_code = prior_class['code']
        prior_sequence = prior_class['sequence']

        # remove artifacts
        seqs = np.delete(seqs, np.where(self.artifact_index>0)[0], axis = 0)

        # apply window size filtering
        seqs = seqs[(seqs[:,3]<self.max_window) & (seqs[:,3]>self.min_window), :]

        # search for location of patterns
        all_stats = np.array(seqs[:,0])



        # search pattern
        # 1. if the last state locked with previous states
        if pattern_sequence[1] == 1:
            locs = np.where((all_stats[:-2] == pattern_code[0]) & (all_stats[1:-1] == pattern_code[1]) & (all_stats[2:] == pattern_code[2]))[0]
            locs = np.int32(locs)

        # 2. if the last state not locked with previous states
        elif pattern_sequence[1] == 0:
            locs = np.where((all_stats[:-1] == pattern_code[0]) & (all_stats[1:] == pattern_code[1]))[0]
            locs_third_temp = np.where(all_stats == pattern_code[2])[0]
            # creating emty space for third pattern locations
            locs_third = []

            # going through for loop and searching third pattern
            for i in range(len(locs)):
                if i+1 < len(locs):
                    index_3 = []
                    index_3 = np.where((locs[i]+1 < locs_third_temp) & (locs[i+1] >= locs_third_temp))[0]
                    if index_3.size:
                        locs_third.append(locs_third_temp[index_3[0]])
                    else:
                        locs_third.append(np.nan)
                else:
                    index_3 = []
                    index_3 = np.where((locs[i]+1 < locs_third_temp))[0]
                    if index_3.size:
                        locs_third.append(locs_third_temp[index_3[0]])
                    else:
                        locs_third.append(np.nan)
            
            # if nan happens remove from both locs and locs_third
            locs = np.delete(locs, np.where(np.isnan(locs_third))[0]).astype(int)
            locs_third = np.delete(locs_third, np.where(np.isnan(locs_third))[0]).astype(int)

        else:
            raise Exception(f'given pattern code is wrong. It has to start with 1 and second digit can only 0 or 1 but {pattern_sequence[1]} is given')
            
        if locs.shape[0] == 0:
            raise Exception('there is no such a pattern in Hypno')

        # search pattern with regards to prior class
        # 1. if prior class locked to the main pattern
        if prior_sequence[0] == 1:
            prior_locs_temp = np.where((all_stats[:-1] == prior_code[0]) & (all_stats[1:] == pattern_code[0]))[0]
            prior_locs_temp = np.array(prior_locs_temp)
            prior_locs = []
            for i in range(len(locs)):
                if i == 0:
                    srch = []
                    srch = np.where(prior_locs_temp < locs[i])[0]
                    if srch.size:
                        prior_locs.append(prior_locs_temp[srch[-1]])
                elif i>0 :
                    srch = []
                    srch = np.where((prior_locs_temp < np.array(locs[i])) & (prior_locs_temp > locs[i-1]))[0]
                    if srch.size:
                        prior_locs.append(prior_locs_temp[srch[-1]])

            if len(prior_locs) == 0:
                raise Exception('There is not pattern of prior to main pattern')

        if prior_sequence[0] == 0:
            prior_locs_temp = np.where(all_stats==prior_code[0])[0]
            prior_locs = []
            if pattern_sequence[1] == 1:
                for i in range(len(locs)):
                    if i == 0:
                        srch = []
                        srch = np.where(prior_locs_temp < locs[i])[0]
                        if srch.size:
                            prior_locs.append(prior_locs_temp[srch[-1]])
                    elif i>0 :
                        srch = []
                        srch = np.where((prior_locs_temp < locs[i]) & (prior_locs_temp > locs[i-1]))[0]
                        if srch.size:
                            prior_locs.append(prior_locs_temp[srch[-1]])


            elif pattern_sequence[1] == 0:
                for i in range(len(locs)):
                    if i == 0:
                        srch = []
                        srch = np.where(prior_locs_temp < locs[i])[0]
                        if srch.size:
                            prior_locs.append(prior_locs_temp[srch[-1]])
                    elif i>0 :
                        srch = []
                        srch = np.where((prior_locs_temp < locs[i]) & (prior_locs_temp > locs_third[i-1]))[0]
                        if srch.size:
                            prior_locs.append(prior_locs_temp[srch[-1]])
                

            
        #cleaning step
        clean_loc = []; clean_third = []
        for i in range(len(prior_locs)):
            if pattern_sequence[1]==1:
                clean_loc.append(np.where(np.array(locs)>prior_locs[i])[0][0])
            elif pattern_sequence[1] == 0:
                clean_loc.append(np.where(np.array(locs)>prior_locs[i])[0][0])
                temp = []; temp = np.where(np.array(locs)>prior_locs[i])[0][0]
                clean_third.append(np.where(np.array(locs_third)>locs[temp])[0][0])

        
        # replacing 
        locs = locs[clean_loc]
        locs_third = locs_third[clean_third]

        print('First pattern locations', locs)
        if pattern_sequence[1] == 0:
            print('Third pattern locations', locs_third)
        print('Prior class location', prior_locs)


        avg_fr = np.zeros((len(locs), 4 * self.pattern['divid']))
        avg_amp = np.zeros((len(locs), 4 * self.pattern['divid']))

        if self.normalize:
            # need to be code
            print('under construction :))')
        else:

            for i in range(len(locs)):
                if pattern_sequence[1] == 1:
                    intervals0 = np.linspace(seqs[prior_locs[i],1], seqs[prior_locs[i],2], self.pattern['divid'] + 1, dtype= int)
                    intervals1 = np.linspace(seqs[locs[i],1], seqs[locs[i],2], self.pattern['divid'] + 1, dtype= int)
                    intervals2 = np.linspace(seqs[locs[i] + 1,1], seqs[locs[i] + 1,2], self.pattern['divid'] + 1, dtype= int)
                    intervals3 = np.linspace(seqs[locs[i] + 2,1], seqs[locs[i] + 2,2], self.pattern['divid'] + 1, dtype= int)
                elif pattern_sequence[1] == 0:
                    intervals0 = np.linspace(seqs[prior_locs[i],1], seqs[prior_locs[i],2], self.pattern['divid'] + 1, dtype= int)
                    intervals1 = np.linspace(seqs[locs[i],1], seqs[locs[i],2], self.pattern['divid'] + 1, dtype= int)
                    intervals2 = np.linspace(seqs[locs[i] + 1,1], seqs[locs[i] + 1,2], self.pattern['divid'] + 1, dtype= int)
                    intervals3 = np.linspace(seqs[locs_third[i],1], seqs[locs_third[i],2], self.pattern['divid'] + 1, dtype= int)

                for ii in range(len(intervals1) -1):
                    
                    if subset['Type'].lower() == 'all':

                        temp_data0 = []; temp_data1 = []; temp_data2 = []; temp_data3 = []
                        temp_data0 = data[intervals0[ii]:intervals0[ii+1],:]
                        temp_data1 = data[intervals1[ii]:intervals1[ii+1],:]
                        temp_data2 = data[intervals2[ii]:intervals2[ii+1],:]
                        temp_data3 = data[intervals3[ii]:intervals3[ii+1],:]

                    elif subset['Type'].lower() == 'top':

                        temp_data0 = []; temp_loc0 = []; temp_ = []; temp_index = []; prct = []
                        temp_data0 = data[intervals0[0]:intervals0[-1],:]
                        temp_loc0 = np.where(temp_data0>0, 1, 0)
                        temp_ = np.sum(temp_loc0, axis = 0) / (temp_data0.shape[0]/sf)  #normalize to length in sec for each cell

                        temp_index = np.argsort(temp_) # sort low to high 0 ... inf
                        prct = subset['Percentage']; cell_nr = np.int(temp_data0.shape[1] * prct)

                        # selecting top xx percent and checking if it doesn't have 0 inside
                        top_xx_index = []
                        
                        for qq in range(temp_data0.shape[1]):
                            if (temp_[temp_index[-(qq+1)]] != 0) and len(top_xx_index)<= cell_nr:
                                top_xx_index.append(temp_index[-(qq+1)])
                        
                                

                        temp_data0 = []; temp_data1 = []; temp_data2 = []; temp_data3 = []
                        temp_data0 = data[intervals0[ii]:intervals0[ii+1],top_xx_index]
                        temp_data1 = data[intervals1[ii]:intervals1[ii+1],top_xx_index]
                        temp_data2 = data[intervals2[ii]:intervals2[ii+1],top_xx_index]
                        temp_data3 = data[intervals3[ii]:intervals3[ii+1],top_xx_index]

                    elif subset['Type'].lower() == 'least':

                        temp_data0 = []; temp_loc0 = []; temp_ = []; temp_index = []; prct = []
                        temp_data0 = data[intervals0[0]:intervals0[-1],:]
                        temp_loc0 = np.where(temp_data0>0, 1, 0)
                        temp_ = np.sum(temp_loc0, axis = 0) / (temp_data0.shape[0]/sf)  #normalize to length in sec for each cell

                        temp_index = np.argsort(temp_) # sort low to high 0 ... inf
                        prct = subset['Percentage']; cell_nr = np.int(temp_data0.shape[1] * prct)

                        # selecting top xx percent and checking if it doesn't have 0 inside
                        least_xx_index = []
                        for qq in range(temp_data0.shape[1]):
                            if (temp_[temp_index[qq]] != 0) and len(least_xx_index)<= cell_nr:
                                least_xx_index.append(temp_index[qq])

                        temp_data0 = []; temp_data1 = []; temp_data2 = []; temp_data3 = []
                        temp_data0 = data[intervals0[ii]:intervals0[ii+1],least_xx_index]
                        temp_data1 = data[intervals1[ii]:intervals1[ii+1],least_xx_index]
                        temp_data2 = data[intervals2[ii]:intervals2[ii+1],least_xx_index]
                        temp_data3 = data[intervals3[ii]:intervals3[ii+1],least_xx_index]

                    elif subset['Type'].lower() == 'nonactive':

                        temp_data0 = []; temp_loc0 = []; temp_ = [];  prct = []
                        temp_data0 = data[intervals0[0]:intervals0[-1],:]
                        temp_loc0 = np.where(temp_data0>0, 1, 0)
                        temp_ = np.sum(temp_loc0, axis = 0) / (temp_data0.shape[0]/sf)  #normalize to length in sec for each cell

                        prct = subset['Percentage']; cell_nr = np.int(temp_data0.shape[1] * prct)

                        non_active_index = []
                        non_active_index = np.where(temp_ == 0)[0]

                        temp_data0 = []; temp_data1 = []; temp_data2 = []; temp_data3 = []
                        temp_data0 = data[intervals0[ii]:intervals0[ii+1],non_active_index]
                        temp_data1 = data[intervals1[ii]:intervals1[ii+1],non_active_index]
                        temp_data2 = data[intervals2[ii]:intervals2[ii+1],non_active_index]
                        temp_data3 = data[intervals3[ii]:intervals3[ii+1],non_active_index]







                    temp_loc0 = []; temp_loc1 = []; temp_loc2 = []; temp_loc3 = []
                    temp_loc0 = np.where(temp_data0>0)
                    temp_loc1 = np.where(temp_data1>0)
                    temp_loc2 = np.where(temp_data2>0)
                    temp_loc3 = np.where(temp_data3>0)

                    avg_amp[i,ii] = temp_data0[temp_loc0[0], temp_loc0[1]].mean()
                    avg_amp[i,ii + self.pattern['divid']] = temp_data1[temp_loc1[0], temp_loc1[1]].mean()
                    avg_amp[i,ii + 2 * self.pattern['divid']] = temp_data2[temp_loc2[0], temp_loc2[1]].mean()
                    avg_amp[i,ii + 3 * self.pattern['divid']] = temp_data3[temp_loc3[0], temp_loc3[1]].mean()


                    avg_fr[i,ii] = len(temp_loc0[0]) / (temp_data0.shape[0]/sf)  #normalize to length in sec
                    avg_fr[i,ii + self.pattern['divid']] = len(temp_loc1[0]) / (temp_data1.shape[0]/sf)  #normalize to length in sec
                    avg_fr[i,ii + 2 * self.pattern['divid']] = len(temp_loc2[0]) / (temp_data2.shape[0]/sf)  #normalize to length in sec
                    avg_fr[i,ii + 3 * self.pattern['divid']] = len(temp_loc3[0]) / (temp_data3.shape[0]/sf)  #normalize to length in sec


        self.type3_prior_class_amp = avg_amp
        self.type3_prior_class_fr = avg_fr

        return avg_amp, avg_fr

    
    # plotting section
    def plot_type1(self, data_amp, data_fr, divid_, pattern_names, ylim_fr, ylim_amp, figsize_, save_address):
        # loading results and preparing dataframe
        
        if not isinstance(data_amp, bool):
            avg_amp = data_amp 
        else:
            avg_amp = self.type1_amp

        if not isinstance(data_fr, bool):
            avg_fr = data_fr 
        else:
            avg_fr = self.type1_fr 

        if not isinstance(divid_, bool):
            nr_divid = divid_ 
        else:
            nr_divid = self.pattern['divid']

        if len(pattern_names) != len(set([word.lower() for word in pattern_names])):
            raise Exception('There is duplicate in the given names. Names need to be unequal.')

        if len(pattern_names) != 1:
            raise Exception('Only one pattern name you can give.')


        # For amplitude
        temp_array = np.zeros((avg_amp.shape[0] * avg_amp.shape[1], 3))
        ro = 0
        for i in range(avg_amp.shape[0]):
            for ii in range(avg_amp.shape[1]):
                temp_array[ro, 0] = avg_amp[i,ii]
                temp_array[ro, 1] = np.mod(ii,nr_divid).astype(int) +1 
                temp_array[ro, 2] = int(ii/nr_divid) + 1
                ro += 1

        DF_amp = pd.DataFrame({'Amplitude (df/f0)': temp_array[:,0], 'level': temp_array[:,1], 'states': temp_array[:,2]})
        DF_amp = DF_amp.replace({'level': {n+1:str(n+1)+'th' for n in range(nr_divid)}, 'states':{1: pattern_names}})


        # For frequency
        temp_array = np.zeros((avg_fr.shape[0] * avg_fr.shape[1], 3))
        ro = 0
        for i in range(avg_fr.shape[0]):
            for ii in range(avg_fr.shape[1]):
                temp_array[ro, 0] = avg_fr[i,ii]
                temp_array[ro, 1] = np.mod(ii,nr_divid).astype(int) +1 
                temp_array[ro, 2] = int(ii/nr_divid) + 1
                ro += 1

        DF_fr = pd.DataFrame({'Frequency (1/sec)': temp_array[:,0], 'level': temp_array[:,1], 'states': temp_array[:,2]})
        DF_fr = DF_fr.replace({'level': {n+1:str(n+1)+'th' for n in range(nr_divid)}, 'states':{1: pattern_names}})

        # plotting violen plot
        fig, (ax0,ax1) = plt.subplots(nrows=1, ncols=2, figsize=figsize_)
        sns.violinplot(data = DF_fr, x = 'states', y = 'Frequency (1/sec)', hue='level', ax=ax0)
        ax0.set_ylim(ylim_fr)

        sns.violinplot(data = DF_amp, x = 'states', y = 'Amplitude (df/f0)', hue='level', ax=ax1)
        ax1.set_ylim(ylim_amp)

        if save_address:
            fig.savefig(save_address + '/TripletAnalysisType1.pdf', dpi=300, format='pdf')




    def plot_type2(self, pattern_names, data_amp, data_fr, divid_, ylim_fr, ylim_amp, figsize_, save_address):
        # loading results and preparing dataframe
        if not isinstance(data_amp, bool):
            avg_amp = data_amp 
        else:
            avg_amp = self.type2_amp

        if not isinstance(data_fr, bool):
            avg_fr = data_fr 
        else:
            avg_fr = self.type2_fr 

        if not isinstance(divid_, bool):
            nr_divid = divid_ 
        else:
            nr_divid = self.pattern['divid']

        if len(pattern_names) != len(set([word.lower() for word in pattern_names])):
            raise Exception('There is duplicate in the given names. Names need to be unequal.')

        if len(pattern_names) != 2:
            raise Exception('Only one pattern name you can give.')


        # For amplitude
        temp_array = np.zeros((avg_amp.shape[0] * avg_amp.shape[1], 3))
        ro = 0
        for i in range(avg_amp.shape[0]):
            for ii in range(avg_amp.shape[1]):
                temp_array[ro, 0] = avg_amp[i,ii]
                temp_array[ro, 1] = np.mod(ii,nr_divid).astype(int) +1 
                temp_array[ro, 2] = int(ii/nr_divid) + 1
                ro += 1

        DF_amp = pd.DataFrame({'Amplitude (df/f0)': temp_array[:,0], 'level': temp_array[:,1], 'states': temp_array[:,2]})
        DF_amp = DF_amp.replace({'level': {n+1:str(n+1)+'th' for n in range(nr_divid)}, 'states':{1: pattern_names[0], 2: pattern_names[1]}})


        # For frequency
        temp_array = np.zeros((avg_fr.shape[0] * avg_fr.shape[1], 3))
        ro = 0
        for i in range(avg_fr.shape[0]):
            for ii in range(avg_fr.shape[1]):
                temp_array[ro, 0] = avg_fr[i,ii]
                temp_array[ro, 1] = np.mod(ii,nr_divid).astype(int) +1 
                temp_array[ro, 2] = int(ii/nr_divid) + 1
                ro += 1

        DF_fr = pd.DataFrame({'Frequency (1/sec)': temp_array[:,0], 'level': temp_array[:,1], 'states': temp_array[:,2]})
        DF_fr = DF_fr.replace({'level': {n+1:str(n+1)+'th' for n in range(nr_divid)}, 'states':{1: pattern_names[0], 2: pattern_names[1]}})

        # plotting violen plot
        fig, (ax0,ax1) = plt.subplots(nrows=1, ncols=2, figsize=figsize_)
        sns.violinplot(data = DF_fr, x = 'states', y = 'Frequency (1/sec)', hue='level', ax=ax0)
        ax0.set_ylim(ylim_fr)

        sns.violinplot(data = DF_amp, x = 'states', y = 'Amplitude (df/f0)', hue='level', ax=ax1)
        ax1.set_ylim(ylim_amp)

        if save_address:
            fig.savefig(save_address + '/TripletAnalysisType2.pdf', dpi=300, format='pdf')

    

    def plot_type3(self, pattern_names, data_amp, data_fr, divid_, ylim_fr, ylim_amp, figsize_, save_address):
        # loading results and preparing dataframe
        if not isinstance(data_amp, bool):
            avg_amp = data_amp 
        else:
            avg_amp = self.type3_amp

        if not isinstance(data_fr, bool):
            avg_fr = data_fr 
        else:
            avg_fr = self.type3_fr 

        if not isinstance(divid_, bool):
            nr_divid = divid_ 
        else:
            nr_divid = self.pattern['divid']

        
        if len(pattern_names) != len(set([word.lower() for word in pattern_names])):
            raise Exception('There is duplicate in the given names. Names need to be unequal.')

        if len(pattern_names) != 3:
            raise Exception('Only one pattern name you can give.')

        

        # For amplitude
        temp_array = np.zeros((avg_amp.shape[0] * avg_amp.shape[1], 3))
        ro = 0
        for i in range(avg_amp.shape[0]):
            for ii in range(avg_amp.shape[1]):
                temp_array[ro, 0] = avg_amp[i,ii]
                temp_array[ro, 1] = np.mod(ii,nr_divid).astype(int) +1 
                temp_array[ro, 2] = int(ii/nr_divid) + 1
                ro += 1

        DF_amp = pd.DataFrame({'Amplitude (df/f0)': temp_array[:,0], 'level': temp_array[:,1], 'states': temp_array[:,2]})
        DF_amp = DF_amp.replace({'level': {n+1:str(n+1)+'th' for n in range(nr_divid)}, 'states':{1: pattern_names[0], 2: pattern_names[1], 3:pattern_names[2]}})


        # For frequency
        temp_array = np.zeros((avg_fr.shape[0] * avg_fr.shape[1], 3))
        ro = 0
        for i in range(avg_fr.shape[0]):
            for ii in range(avg_fr.shape[1]):
                temp_array[ro, 0] = avg_fr[i,ii]
                temp_array[ro, 1] = np.mod(ii,nr_divid).astype(int) +1 
                temp_array[ro, 2] = int(ii/nr_divid) + 1
                ro += 1

        DF_fr = pd.DataFrame({'Frequency (1/sec)': temp_array[:,0], 'level': temp_array[:,1], 'states': temp_array[:,2]})
        DF_fr = DF_fr.replace({'level': {n+1:str(n+1)+'th' for n in range(nr_divid)}, 'states':{1: pattern_names[0], 2: pattern_names[1], 3:pattern_names[2]}})

        # plotting violen plot
        fig, (ax0,ax1) = plt.subplots(nrows=1, ncols=2, figsize=figsize_)
        sns.violinplot(data = DF_fr, x = 'states', y = 'Frequency (1/sec)', hue='level', ax=ax0)
        ax0.set_ylim(ylim_fr)

        sns.violinplot(data = DF_amp, x = 'states', y = 'Amplitude (df/f0)', hue='level', ax=ax1)
        ax1.set_ylim(ylim_amp)

        if save_address:
            fig.savefig(save_address + '/TripletAnalysisType3.pdf', dpi=300, format='pdf')



    def plot_type3_prior_class(self, pattern_names, data_amp, data_fr, divid_, ylim_fr, ylim_amp, figsize_, save_address):
        # loading results and preparing dataframe
        if not isinstance(data_amp, bool):
            avg_amp = data_amp 
        else:
            avg_amp = self.type3_prior_class_amp

        if not isinstance(data_fr, bool):
            avg_fr = data_fr 
        else:
            avg_fr = self.type3_prior_class_fr

        if not isinstance(divid_, bool):
            nr_divid = divid_ 
        else:
            nr_divid = self.pattern['divid']

        
        if len(pattern_names) != len(set([word.lower() for word in pattern_names])):
            raise Exception('There is duplicate in the given names. Names need to be unequal.')

        if len(pattern_names) != 4:
            raise Exception('Only one pattern name you can give.')

        

        # For amplitude
        temp_array = np.zeros((avg_amp.shape[0] * avg_amp.shape[1], 3))
        ro = 0
        for i in range(avg_amp.shape[0]):
            for ii in range(avg_amp.shape[1]):
                temp_array[ro, 0] = avg_amp[i,ii]
                temp_array[ro, 1] = np.mod(ii,nr_divid).astype(int) +1 
                temp_array[ro, 2] = int(ii/nr_divid) + 1
                ro += 1

        DF_amp = pd.DataFrame({'Amplitude (df/f0)': temp_array[:,0], 'level': temp_array[:,1], 'states': temp_array[:,2]})
        DF_amp = DF_amp.replace({'level': {n+1:str(n+1)+'th' for n in range(nr_divid)}, 'states':{1: pattern_names[0], 
                                                                                                  2: pattern_names[1], 
                                                                                                  3:pattern_names[2], 
                                                                                                  4:pattern_names[3]}})


        # For frequency
        temp_array = np.zeros((avg_fr.shape[0] * avg_fr.shape[1], 3))
        ro = 0
        for i in range(avg_fr.shape[0]):
            for ii in range(avg_fr.shape[1]):
                temp_array[ro, 0] = avg_fr[i,ii]
                temp_array[ro, 1] = np.mod(ii,nr_divid).astype(int) +1 
                temp_array[ro, 2] = int(ii/nr_divid) + 1
                ro += 1

        DF_fr = pd.DataFrame({'Frequency (1/sec)': temp_array[:,0], 'level': temp_array[:,1], 'states': temp_array[:,2]})
        DF_fr = DF_fr.replace({'level': {n+1:str(n+1)+'th' for n in range(nr_divid)}, 'states':{1: pattern_names[0], 
                                                                                                2: pattern_names[1], 
                                                                                                3:pattern_names[2], 
                                                                                                4:pattern_names[3]}})

        # plotting violen plot
        fig, (ax0,ax1) = plt.subplots(nrows=1, ncols=2, figsize=figsize_)
        sns.violinplot(data = DF_fr, x = 'states', y = 'Frequency (1/sec)', hue='level', ax=ax0)
        ax0.set_ylim(ylim_fr)

        sns.violinplot(data = DF_amp, x = 'states', y = 'Amplitude (df/f0)', hue='level', ax=ax1)
        ax1.set_ylim(ylim_amp)

        if save_address:
            fig.savefig(save_address + '/TripletAnalysisType3.pdf', dpi=300, format='pdf')


def select_subpopulation(avg_data, type_, percentage_):
    # this function is finding index of of sub-population in a given state
    # user has to give average data of amplitude or frequency for any interested state and this function return just index of cells

    # avg_data --> average amplitude/ average frequency for a given state
    # type_ --> 'top' or 'least'
    # percentage_ --> percentage of subpopulation

    data_length = len(avg_data)
    fraction = np.array((data_length * (percentage_/100)), dtype = int)

    avg_data = np.array(avg_data)

    index = np.argsort(avg_data)

    if type_.lower() == 'top':
        out_index = index[-fraction:]
    elif type_.lower() == 'least':
        out_index = index[:fraction]
    else:
        raise Exception(f'Type must be Top or Least but {type} is given.')

    if np.where(np.isnan(avg_data[out_index]))[0].size:
        print(f'{np.where(np.isnan(avg_data[out_index]))[0].size} NaNs exist in selected indices and will be removed.')

    out_index = np.delete(out_index, np.where(np.isnan(avg_data[out_index]))[0])

    

    return out_index

    