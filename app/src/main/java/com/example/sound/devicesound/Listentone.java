package com.example.sound.devicesound;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.util.Log;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.*;

import java.util.ArrayList;

public class Listentone {

    int HANDSHAKE_START_HZ = 4096;
    int HANDSHAKE_END_HZ = 5120 + 1024;

    int START_HZ = 1024;
    int STEP_HZ = 256;
    int BITS = 4;

    int FEC_BYTES = 4;

    private int mAudioSource = MediaRecorder.AudioSource.MIC;
    private int mSampleRate = 44100; // 샘플링을 위한 샘플링rate 설정
    private int mChannelCount = AudioFormat.CHANNEL_IN_MONO;
    private int mAudioFormat = AudioFormat.ENCODING_PCM_16BIT;
    private float interval = 0.1f;

    private int mBufferSize = AudioRecord.getMinBufferSize(mSampleRate, mChannelCount, mAudioFormat);

    public AudioRecord mAudioRecord = null;
    int audioEncodig;
    boolean startFlag;
    FastFourierTransformer transform;

    public Listentone() {

        transform = new FastFourierTransformer(DftNormalization.STANDARD);
        startFlag = false;
        mAudioRecord = new AudioRecord(mAudioSource, mSampleRate, mChannelCount, mAudioFormat, mBufferSize);
        mAudioRecord.startRecording();
    }

    int blocksize = findPowerSize((int)(long)Math.round(interval/2*mSampleRate));
    short[] buffer = new short[blocksize];

    private int findPowerSize(int number) {
        //가까운 2의 제곱수 찾아야함.
        int n =0;
        while(true){
            if( (Math.pow(2, n+1) >= number) && (number >= Math.pow(2, n)) ) {
                break;
            }
            n++;
        }

        return (int)Math.pow(2, n); //Math.pow 함수는 출력이 실수형이다.
    }

    private int findFrequency(double[] toTransform) {
        int len = toTransform.length;
        double realNum;
        double imgNum;
        double[] mag = new double[len];

        Complex[] complx = transform.transform(toTransform, TransformType.FORWARD);
        //푸리에 변환 결과 복소수가 만들어진다.

        for (int i = 0; i < complx.length; i++) {
            realNum = complx[i].getReal();
            imgNum = complx[i].getImaginary();
            mag[i] = Math.sqrt((realNum * realNum) + (imgNum * imgNum));
        }

        Double[] freq = this.fftfreq(complx.length,1.0);

        //푸리에 변환해서 최대값을 뽑아내야함. np.argmax()
        double peak_coeff = 0;
        int index = 0;
        for(int i = 0; i < complx.length; i++) {
            if(peak_coeff < mag[i]) {
                peak_coeff = mag[i];
                index = i;
            }
        }
        Double peak_freq = freq[index];

        /*for(int i = 0; i< freq.length; i++) {
            Log.d("normalization : ", Double.toString(freq[i]));
        }*/

        return Math.abs((int)(mSampleRate * peak_freq));
    }

    // https://github.com/numpy/numpy/blob/v1.16.1/numpy/fft/helper.py#L131-L177
    /*
     val = 1.0 / (n * d)
    results = empty(n, int)
    N = (n-1)//2 + 1
    p1 = arange(0, N, dtype=int)
    results[:N] = p1
    p2 = arange(-(n//2), 0, dtype=int)
    results[N:] = p2
    return results * val
    * */
    //실수 변환과 정규화(일정하게 값이 변화됨 => 정규분포 변환시키던거처럼!!)를 위한 값들을 지정해준다.
    //정규화를 더 잘할 수 있다면 정확한 글자를 뽑아낼 수 있을것 같은데..
    private Double[] fftfreq(int n, double d) {
        double val = 1.0 / (n * d); // #val = 1.0 / (n * d)
        Double[] results = new Double[n];
        int N = (n - 1) / 2 + 1;
        for(int i = 0; i <= N; i++) {
            results[i] = (double)i;
        }
        int p2 = - (n / 2);
        for(int i = N + 1; i < n ; i++) {
            results[i] = (double)p2;
            p2++;
        }
        for(int i = 0; i < n; i++) {
            results[i] = results[i] * val;
        }
        return results;

        /*for(int i = 0; i < n; i++) {
            results[i] = results[i] * val;
        }*/
    }

    private ArrayList<Integer> extract_packet(ArrayList<Integer> freqs) {
        ArrayList<Integer> temp = new ArrayList<Integer>();
        ArrayList<Integer> bit_chunks = new ArrayList<Integer>();

        for(int i = 0; i < freqs.size(); i++) {
            int t = (int)(Math.round((freqs.get(i) - START_HZ) / STEP_HZ));
            temp.add(t);
        }

        for(int c = 1; c < temp.size(); c++) {
            if((0 <= temp.get(c)) && (Math.pow(2, BITS)) > temp.get(c)) {
                bit_chunks.add(temp.get(c));
            }
        }
        return decode_bitchunks(BITS, bit_chunks);
    }

    //그대로 옮김 : python 코드를 java 코드로~
    //4비트씩 쪼개어주는 역활을 했던 것 같음.
    private ArrayList<Integer> decode_bitchunks(int chunk_bits, ArrayList<Integer> chunks){
        ArrayList<Integer> out_bytes = new ArrayList<Integer>();

        int next_read_chunk = 0;
        int next_read_bit = 0;
        int byte_ = 0;
        int bits_left = 8;

        while(next_read_chunk < chunks.size()) {
            int can_fill = chunk_bits - next_read_bit;
            int to_fill = Math.min(bits_left, can_fill);
            int offset = chunk_bits - next_read_bit - to_fill;
            byte_ <<= to_fill;
            int shifted = chunks.get(next_read_chunk) & (((1 << to_fill) - 1) << offset);
            byte_ |= shifted >> offset;
            bits_left -= to_fill;
            next_read_bit += to_fill;
            if(bits_left <= 0) {
                out_bytes.add(byte_);
                byte_ = 0;
                bits_left = 8;
            }
            if(next_read_bit >= chunk_bits) {
                next_read_chunk += 1;
                next_read_bit -= chunk_bits;
            }
        }
        return out_bytes;
    }

    private boolean match(double freq1, double freq2) {
        return Math.abs(freq1 - freq2) < 20;
    }

    public void PreRequest() {
        ArrayList<Integer> packet = new ArrayList<Integer>();
        double[] chunk = new double[blocksize];

        boolean in_packet = false;
        int dom = 0;

        ArrayList<Integer> byte_stream = new ArrayList<Integer>();

        while (true) {
            int bufferedReadResult = mAudioRecord.read(buffer,0,blocksize);

            if(bufferedReadResult < 0) continue; // if not l: continue

            for(int i = 0; i < blocksize; i++){
                chunk[i] = (double)buffer[i];
            }

            dom = this.findFrequency(chunk);

            if (in_packet && match(dom, HANDSHAKE_END_HZ)) {
                byte_stream = extract_packet(packet);
                String result = "";

                for(int i = 0; i < byte_stream.size(); i++) {
                    int temp = byte_stream.get(i);
                    result = result + ((char)temp);
                }
                Log.d("ListenTone", result);
                packet.clear();
                in_packet = false;
            }
            else if(in_packet) {
                packet.add(dom);
            }
            else if (match(dom, HANDSHAKE_START_HZ)) {
                in_packet = true;
            }
        }
    }
}