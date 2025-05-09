

import numpy as np
import mne


class CHBEdfFile(object):
    """
    Edf reader using pyedflib
    """

    def __init__(self, filename, patient_id=None, ch_num=1, doing_lowpass_filter=False, preload=False):
        self._filename = filename
        self._patient_id = patient_id
        self.ch_num = ch_num
        self._raw_data = mne.io.read_raw_edf(filename, preload=preload)
        self._info = self._raw_data.info
        self.doing_lowpass_filter = doing_lowpass_filter

    def get_filepath(self):
        """
        Name of the EDF path
        """
        return self._filename

    def get_filename(self):
        """
        Name of the EDF name
        """
        return self._filename.split("/")[-1].split(".")[0]

    def get_n_channels(self):
        """
        Number of channels
        """
        return self._info['nchan']

    def get_n_data_points(self):  # 3686400
        """
        Number of data points
        """
        return len(self._raw_data._times)

    def get_channel_names(self):
        """
        Names of channels
        """
        return self._info['ch_names']

    def get_file_duration(self):  # 3600
        """
        Returns the file duration in seconds
        """
        return int(round(self._raw_data._last_time))

    def get_sampling_rate(self):  # 1024
        """
        Get the frequency
        """
        if self._info['sfreq'] < 1:
            raise ValueError("sampling frequency is less than 1")
        return int(self._info['sfreq'])

    def get_preprocessed_data(self):
        """
        Get preprocessed data
        """
        # sampling frequency
        sfreq = self.get_sampling_rate()  # 256 CHB
        # data loading from edf files
        self._raw_data.load_data()
        # channel selection
        self._raw_data.pick_channels(self.get_pick_channels())  # 18chs
        # filter
        if self.doing_lowpass_filter:
            self._raw_data.filter(0, 64, n_jobs=16)  # 0-64Hz
            self._raw_data.notch_filter(np.arange(60, int((sfreq / 2) // 60 * 60 + 1), 60), n_jobs=16)  # (channel,sample)
        # resample
        if sfreq > 256:
            self._raw_data.resample(256, n_jobs=16)
            # self._raw_data.resample(256,n_jobs=16)
        data = self._raw_data.get_data().transpose(1, 0)  # (sample,channel)
        return data

    def get_pick_channels(self):
        """
        Get used channel names
        for CHB, use the common 18 channels
        """
        pick_channels = []

        # 23/28 chs -> 18chs
        if self._patient_id in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 20, 21, 22, 23]:
            pick_channels = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3',
                             'C3-P3', 'P3-O1', 'FZ-CZ', 'CZ-PZ', 'FP2-F4', 'F4-C4',
                             'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2']

        # 28 25 22 28 22 28 chs -> 18chs
        elif self._patient_id in [13, 16, 17, 18, 19]:

            if self.get_n_channels() == 28:
                pick_channels = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3',
                                 'C3-P3', 'P3-O1', 'FZ-CZ', 'CZ-PZ', 'FP2-F4', 'F4-C4',
                                 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2']

            else:  # 22/25
                pick_channels = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3',
                                 'C3-P3', 'P3-O1', 'FZ-CZ', 'CZ-PZ', 'FP2-F4', 'F4-C4',
                                 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2']

        return pick_channels

    def plot_signal(self, duration, n_channels):
        '''
        plot signals
        '''
        self._raw_data.plot(duration=duration, n_channels=n_channels)


class XWbdfFile(object):
    """
    Edf reader using pyedflib
    """

    def __init__(self, filename, patient_id=None, ch_num=1, doing_lowpass_filter=True, preload=False):
        self._filename = filename
        self._patient_id = patient_id
        self.ch_num = ch_num
        self._raw_data = mne.io.read_raw_bdf(filename, preload=preload)
        self._info = self._raw_data.info
        self.doing_lowpass_filter = doing_lowpass_filter

    def get_filepath(self):
        """
        Name of the EDF path
        """
        return self._filename

    def get_filename(self):
        """
        Name of the EDF name
        """
        return self._filename.split("/")[-1].split(".")[0]

    def get_n_channels(self):
        """
        Number of channels
        """
        return self._info['nchan']

    def get_n_data_points(self):  # 3686400
        """
        Number of data points
        """
        return len(self._raw_data._times)

    def get_channel_names(self):
        """
        Names of channels
        """
        return self._info['ch_names']

    def get_file_duration(self):  # 3600
        """
        Returns the file duration in seconds
        """
        return int(round(self._raw_data._last_time))

    def get_sampling_rate(self):  # 1024
        """
        Get the frequency
        """
        if self._info['sfreq'] < 1:
            raise ValueError("sampling frequency is less than 1")
        return int(self._info['sfreq'])

    def get_preprocessed_data(self):
        """
        Get preprocessed data
        """
        # sampling frequency
        sfreq = self.get_sampling_rate()  # 256 CHB
        print("orisfreq", sfreq)
        # data loading from edf files
        self._raw_data.load_data()
        # channel selection
        self._raw_data.pick_channels(self.get_pick_channels())  # 18chs
        # filter
        if self.doing_lowpass_filter:
            self._raw_data.filter(0, 64, n_jobs=16)  # 0-64Hz
            self._raw_data.notch_filter(np.arange(60, int((sfreq / 2) // 60 * 60 + 1), 60), n_jobs=16)  # (channel,sample)
        # resample
        if sfreq > 256:
            # self._raw_data.resample(512)
            self._raw_data.resample(512, n_jobs=16)
        data = self._raw_data.get_data().transpose(1, 0)  # (sample,channel)
        return data

    def get_pick_channels(self):
        """
        Get used channel names
        for CHB, use the common 18 channels
        """
        # pick_channels = []
        # pick_channels = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'B1', 'B2', 'B3', 'B4', 'B5',
        #                  'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12',
        #                  'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18',
        #                  'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'G1', 'G2',
        #                  'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9',
        #                  'H10', 'H11', 'H12']

        pick_channels = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'C1', 'C2', 'C3', 'C4',
                         'C5', 'C6', 'C7',
                         'C8', 'C9', 'C10', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7',
                         'E8', 'E9',
                         'E10', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10']
        # 70

        ############################
        # if self._patient_id in [1, 100]:
        #     pick_channels = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'B1', 'B2',
        #                      'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
        #                      'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9',
        #                      'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11',
        #                      'E12', 'E13', 'E14', 'E15', 'E16', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'G1',
        #                      'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8',
        #                      'H9', 'H10', 'H11', 'H12', 'H13', 'H14', 'H15', 'H16', 'H17', 'H18', 'ECG1']
        # elif self._patient_id in [2, 200]:
        #     pick_channels = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'B1', 'B2', 'B3', 'B4',
        #                      'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
        #                      'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12',
        #                      'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'F1', 'F2', 'F3', 'F4',
        #                      'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14',
        #                      'G15', 'G16', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'ECG1']
        # elif self._patient_id in [3, 300]:
        #     pick_channels = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20',
        #                      'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
        #                      'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9',
        #                      'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11',
        #                      'E12', 'E13', 'E14', 'E15', 'E16', 'E17', 'E18', 'E19', 'E20', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11',
        #                      'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11',
        #                      'G12', 'G13', 'G14', 'G15', 'G16', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H13', 'H14', 'H15',
        #                      'H16', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13', 'I14', 'I15', 'I16', 'J1', 'J2', 'J3',
        #                      'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'K1', 'K2', 'K3', 'K4', 'K5', 'K6', 'K7', 'K8', 'K9', 'K10',
        #                      'K11', 'K12', 'K13', 'K14', 'K15', 'K16', 'ECG']
        ############################

        return pick_channels

    def plot_signal(self, duration, n_channels):
        '''
        plot signals
        '''
        self._raw_data.plot(duration=duration, n_channels=n_channels)
