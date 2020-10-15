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
# Just read files in h5 format





class ETL:
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

    def mean_fire(self, file_nr, sr, amplitude_NaN_Zero):
        """
        mean firing rate is average of events normalized to length of input data per each individual cell
        """
        hypno = self.hypno
        data = self.data
        hypno_artifacts = self.hypno_artifacts
        data_artifacts = self.data_artifacts
        
        if  data.shape == data_artifacts.shape and hypno.shape == hypno_artifacts.shape:
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
        
        if data.shape == data_artifacts.shape and hypno.shape == hypno_artifacts.shape:
            data = data * np.logical_not(data_artifacts)    
            hypno = hypno * np.logical_not(hypno_artifacts)

        
        # calculating mean firing rate
        data_sws = [np.where(np.where(hypno == -2 , 1, 0), data[:,i], 0) for i in range(data.shape[1])]
        data_rem = [np.where(np.where(hypno == -3 , 1, 0), data[:,i], 0) for i in range(data.shape[1])]
        data_awake = [np.where(np.where(hypno == 0 , 1, 0), data[:,i], 0) for i in range(data.shape[1])]

        data_sws = np.array(data_sws).T
        data_rem = np.array(data_rem).T
        data_awake = np.array(data_awake).T

        mean_fr_sws = np.sum(np.where(data_sws > 0 , 1, 0), axis = 0) / ((np.sum(np.where(hypno == -2 , 1, 0)) - artifacts_length_sws) / sr)
        mean_fr_rem = np.sum(np.where(data_rem > 0 , 1, 0), axis = 0) / ((np.sum(np.where(hypno == -3 , 1, 0)) - artifacts_length_rem) / sr)
        mean_fr_awake = np.sum(np.where(data_awake > 0 , 1, 0), axis = 0) / ((np.sum(np.where(hypno == 0 , 1, 0)) - artifacts_length_awake) / sr)

        # claculating mean amplitude

        mean_amp_sws = np.sum(np.where(data_sws > 0, data_sws, 0), axis = 0) / np.sum(np.where(data_sws > 0, 1, 0), axis = 0)
        mean_amp_rem = np.sum(np.where(data_rem > 0, data_rem, 0), axis = 0) / np.sum(np.where(data_rem > 0, 1, 0), axis = 0)
        mean_amp_awake = np.sum(np.where(data_awake > 0, data_awake, 0), axis = 0) / np.sum(np.where(data_awake > 0, 1, 0), axis = 0)

        if amplitude_NaN_Zero:

            print('*'*100)
            print('*'*100)
            print('**'+ ' '*45 + 'WARNING' +' '*44 + '**')
            print('**'+ ' ' + 'To calculate average frequency in all Hypno categories if any of cells dont show any activity ' +' ' + '**')
            print('**'+ ' '*1 + ', then mean frequency of those cells are zero (0). However this is different in the case of' +' '*4 + '**')
            print('**'+ ' '*2 + 'amplitude calculation and the result of mean amplitude for non-active cells is NAN value. But' +' '*1 + '**')
            print('**'+ ' '*2 + 'keeping two different numbers could be misleading in distribution representation. To solve' +' '*4 + '**')
            print('**'+ ' '*2 + 'this problem NAN values in mean amplitude calculation are artificially replaced by zeros.' +' '*5 + '**')
            print('*'*100)
            print('*'*100)

            # changing nan values to zeros
            mean_amp_sws = np.where(np.isnan(mean_amp_sws), 0, mean_amp_sws)    
            mean_amp_rem = np.where(np.isnan(mean_amp_rem), 0, mean_amp_rem)
            mean_amp_awake = np.where(np.isnan(mean_amp_awake), 0, mean_amp_awake)

        
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


def run_per_mouse(sf_, ending_, unit_lenght_, address_, set_artifacts_, type_, amplitude_NaN_Zero):
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
        for qq, qq_data in enumerate(set_artifacts_):
            ETL_.set_artifact(start_= np.array(qq_data[0]), end_= np.array(qq_data[1]), file_= np.array(qq_data[2]))
            print(f'setting artifacts for file {np.array(qq_data[2])}')


    # getting artifacts
    data_artifacts, hypno_artifacts = ETL_.get_artifacts()
    print(f'data artifact shape {data_artifacts.shape}, hypno_artifact shape {hypno_artifacts.shape}')


    # initializing activation class
    if not(set_artifacts_):
        activation_ = activation(hypno=Hypno_, data=full_dataset, hypno_artifacts=np.empty(1), data_artifacts=np.empty(1))
    else:
        activation_ = activation(hypno=Hypno_, data=full_dataset, hypno_artifacts=hypno_artifacts, data_artifacts=data_artifacts)

    frequency_ = []
    amplitude_ = []

    for i in range(Hypno_.shape[1]):
        
        # executing mean firing rate calculation
        activation_.mean_fire(file_nr=i, sr=sf_, amplitude_NaN_Zero = amplitude_NaN_Zero)


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


def run_per_mouse_concatinated(sf_, ending_, unit_lenght_, address_, set_artifacts_, amplitude_NaN_Zero):

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
        for qq, qq_data in enumerate(set_artifacts_):
            ETL_.set_artifact(start_= np.array(qq_data[0]), end_= np.array(qq_data[1]), file_= np.array(qq_data[2]))
            print(f'setting artifacts for file {np.array(qq_data[2])}')


    # getting artifacts
    data_artifacts, hypno_artifacts = ETL_.get_artifacts()
    print(f'data artifact shape {data_artifacts.shape}, hypno_artifact shape {hypno_artifacts.shape}')


    # initializing activation class
    if not(set_artifacts_):
        activation_ = activation(hypno=np.hstack(Hypno_)[:,np.newaxis], data=np.hstack(full_dataset.T).T[:,:,np.newaxis], 
                         hypno_artifacts=np.empty(1), data_artifacts=np.empty(1))
    else:
        activation_ = activation(hypno=np.hstack(Hypno_)[:,np.newaxis], data=np.hstack(full_dataset.T).T[:,:,np.newaxis], 
                             hypno_artifacts=np.hstack(hypno_artifacts)[:,np.newaxis], 
                             data_artifacts=np.hstack(data_artifacts.T).T[:,:,np.newaxis])

    activation_.mean_fire(file_nr=0, sr=sf_, amplitude_NaN_Zero = amplitude_NaN_Zero)
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

        XY_select = selection(XY, limitXY=range_)


        xAmplitudes = np.array(XY_select)[:,0]#your data here
        yAmplitudes = np.array(XY_select)[:,1]#your other data here


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


    def join_plot(self, bins_, kind_):

        x = self.x
        y = self.y
        save_address = self.save_address

        plt.figure(figsize = (10,10))
        g = sns.jointplot(x=x, y=y, kind=kind_, color="#4CB391",
        marginal_kws=dict(bins=bins_, fill=True))
        g.set_axis_labels(xlabel=self.xlabel, ylabel=self.ylabel)
        range__ = [bins_[0], bins_[-1]]
        plt.plot(range__, range__, ls = '--', c = 'k')
        if save_address:
            g.savefig(save_address + '/Joint plot.pdf', dpi=300, format='pdf')
        


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
        
