
import torch
import torch.nn as nn
import librosa
import numpy as np
from stft import (LogmelFilterBank, spectrogram_STFTInput)
import math

def nCr(n, r):
    return math.factorial(n) // math.factorial(r) // math.factorial(n-r)


class LogmelDirCue_Extractor(nn.Module):
    #MAY NEED TO CHANGE THIS FOR BINAURAL
    def __init__(self):
        super().__init__()#checked2

    
        sample_rate, n_fft, hop_length, window, n_mels = \
            24000, 960, 480, "hann", 64
        win_length = 2 * hop_length

        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.n_mels=n_mels
        self.nfft=n_fft
        # STFT extractor

        self.hopsize=hop_length
        fft_window = librosa.filters.get_window(window, n_fft, fftbins=True)#checked3
        
        self.window = torch.from_numpy(librosa.util.pad_center(fft_window, size=n_fft)).to(device="cuda")#checked3
        # Spectrogram extractor
        self.spectrogram_extractor = spectrogram_STFTInput
        
        # Logmel extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=n_fft, 
            n_mels=n_mels, fmin=20, fmax=sample_rate/2, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)
        # Intensity vector extractor
 
        
        self.melW = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels,
            fmin=20, fmax=sample_rate/2).T#checked2
        # (n_fft // 2 + 1, mel_bins)
        self.melW = torch.Tensor(self.melW)#checked2
      
        self.melW=self.melW.to(dtype=torch.complex128)   #checked2
       
        self.melW = nn.Parameter(self.melW)#checked2
        self.melW.requires_grad = False #checked2


    def forward(self, x):
        """
        input: 
            (batch_size, channels=4, data_length)
        output: 
            (batch_size, channels, time_steps, freq_bins) freq_bins->mel_bins
        """
        x=torch.from_numpy(x)
        x=x.view(1,x.shape[0], x.shape[1])#checked3
        x=x.to(device="cuda")#checked3
        if x.ndim != 3:
            raise ValueError("x shape must be (batch_size, num_channels, data_length)\n \
                            Now it is {}".format(x.shape))
        
        x_00=x[:,0,:]#checked3
        x_01=x[:,1,:]#checked3
        dev=x_00.get_device()#checked3
        self.window=self.window.to(device=("cuda:"+str(dev)))#checked3
        Px = torch.stft(input=x_00,
                        n_fft=self.nfft,
                        hop_length=self.hopsize,
                        win_length=self.nfft,
                        window=self.window,
                        center=True,
                        pad_mode='reflect',
                        normalized=False, onesided=None, return_complex=True)#checked3
        Px=torch.transpose(Px,1,2)#checked3
        dev=x_01.get_device()#checked3
        self.window=self.window.to(device=("cuda:"+str(dev)))#checked3
        Px_ref = torch.stft(input=x_01,
                            n_fft=self.nfft,
                            win_length=self.nfft,
                            hop_length=self.hopsize,
                            window=self.window,
                            center=True,
                            pad_mode='reflect',
                            normalized=False, onesided=None, return_complex=True)#checked3
        Px_ref=torch.transpose(Px_ref,1,2)#checked3
        x_0=Px#checked3
        x_1=Px_ref#checked3
        x_0raw=x_0#checked3
        x_1raw=x_1#checked3
        x_0rawmel=torch.matmul(x_0raw, self.melW)#checked3
        x_1rawmel=torch.matmul(x_1raw, self.melW)#checked3
        a1=torch.angle(x_0rawmel)#checked3
        a2=torch.angle(x_1rawmel)#checked3
        sinipd=torch.sin(a1-a2)#checked3
        cosipd=torch.cos(a1-a2)#checked3
     
        (a,c,d)=x_0.shape#checked3
        x_0=x_0.view(a,1,c,d)#checked3
        x_1=x_1.view(a,1,c,d)#checked3
        xtemp1=torch.cat((x_0.real,x_1.real),dim=1)#checked3
        xtemp2=torch.cat((x_0.imag,x_1.imag),dim=1)#checked3
        x=(xtemp1,xtemp2)#checked3
        
        raw_spec,logmel = self.logmel_extractor(self.spectrogram_extractor(x).to(dtype=torch.float))#checked3
        
        value = 1e-20#checked3
        ild=raw_spec[:,0,:,:]/(raw_spec[:,1,:,:]+value)#checked3

        (a,b,c,d)=logmel.shape#checked3
        

            
          
        ild=ild.view(a,1,c,d)#checked3
        sinipd=sinipd.view(a,1,c,d)#checked2
        cosipd=cosipd.view(a,1,c,d)#checked2
        
        # print(temp.shape)
        #GCC Features
        R = x_0raw*torch.conj(x_1raw)#checked3
        gcc = torch.fft.irfft(torch.exp(1.j*torch.angle(R)))#checked3
        gcc = torch.cat((gcc[:,:,-self.n_mels//2:],gcc[:,:,:self.n_mels//2]),dim=-1)#checked3
        gcc = gcc.view(a,1,c,d)#checked3   
        out = torch.cat((logmel, ild,sinipd,cosipd,gcc), dim=1)#checked3
        #out = torch.cat((logmel,gcc), dim=1)#checked3
        out=out.float()#checked3
        
        out=out.view(out.shape[1],out.shape[2],out.shape[3])
       
        return out#checked3