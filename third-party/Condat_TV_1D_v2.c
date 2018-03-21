/*
Total variation denoising of 1-D signals, a.k.a. Fused lasso
signal approximator, by Laurent Condat.

Version 2.0, Aug. 30, 2017.

Given a real vector y of length N and a real lambda>=0, the 
goal is to compute the real vector x minimizing 
    ||x-y||_2^2/2 + lambda.TV(x), 
where ||x-y||_2^2 = sum_{n=1}^{N} (x[n]-y[n])^2 and
TV(x) = sum_{n=1}^{N-1} |x[n+1]-x[n]|.

I proposed a fast and exact algorithm (say, the version 1.0)
for this problem in L. Condat, "A direct algorithm for 1D
total variation denoising," IEEE Signal Proc. Letters, vol.
20, no. 11, pp. 1054-1057, Nov. 2013.
It has worst case complexity O(N^2) but it is recognized as
the fastest in practice (using the C code on my webpage).

The present code is a C implementation of a NEW algorithm, 
which combines the advantages of the v1 algorithm with the 
optimal O(N) complexity of the taut string algorithm.
That is, it is exact, numerically robust (averages of values
computed by Welford-Knuth running mean algorithm, not by sum
divided by length), roughly as fast as the v1 algorithm, and 
it has linear time complexity.
Speed: the computation time of this C code is typically 
85%-120% of the computation time of the C code of the v1. 

In a nutshell, the algorithm is based on the classical Pool
Adjacent Violators Algorithm for isotonic regression, to 
maintain two nonincreasing and nondecreasing (instead of 
constant in the v1) lower and upper approximations of the
signal.

If lambda=0, the algorithm returns x=y, but not exactly, only
up to machine precision.

Usage rights : Copyright Laurent Condat.
This file is distributed under the terms of the CeCILL
licence (compatible with the GNU GPL), which can be
found at the URL "http://www.cecill.info".
*/

#define datatype float
/* double is recommended; in that case the precision is of order 1e-11. With float, the precision for a typical signal with values in [0,255] is of order 5e-3 only. */


/*
In the notations above, input is y, output is x, width is N.
We must have width>=1 and lambda>=0.
The initial content of the output buffer does not matter.
The buffers input and output must be allocated of size at least width.
The algorithm can operate in place, with output=input; in that case the input is replaced by the output.

See the Matlab code on my webpage for comments.
*/
void TV1D_denoise_v2(datatype* input, datatype* output, unsigned int width, const datatype lambda) {
	
	unsigned int* indstart_low = (unsigned int*)malloc(sizeof *indstart_low * width);
	unsigned int* indstart_up = (unsigned int*)malloc(sizeof *indstart_up * width);
	unsigned int j_low = 0, j_up = 0, jseg = 0, indjseg = 0, i=1, indjseg2, ind;
	double output_low_first = input[0]-lambda;
	double output_low_curr = output_low_first;
	double output_up_first = input[0]+lambda;
	double output_up_curr = output_up_first;
	const double twolambda=2.0*lambda;
	if (width==1) {output[0]=input[0]; return;}
	indstart_low[0] = 0;
	indstart_up[0] = 0;
	width--;
	for (; i<width; i++) {
	    if (input[i]>=output_low_curr) {
	    	if (input[i]<=output_up_curr) {
		        output_up_curr+=(input[i]-output_up_curr)/(i-indstart_up[j_up]+1);
		        output[indjseg]=output_up_first;
		        while ((j_up>jseg)&&(output_up_curr<=output[ind=indstart_up[j_up-1]]))
		        	output_up_curr+=(output[ind]-output_up_curr)*
		        		((double)(indstart_up[j_up--]-ind)/(i-ind+1));
		        if (j_up==jseg) {
			        while ((output_up_curr<=output_low_first)&&(jseg<j_low)) {
			        	indjseg2=indstart_low[++jseg];
				    	output_up_curr+=(output_up_curr-output_low_first)*
				    		((double)(indjseg2-indjseg)/(i-indjseg2+1));
				    	while (indjseg<indjseg2) output[indjseg++]=output_low_first;
				    	output_low_first=output[indjseg];
			        }
	    			output_up_first=output_up_curr;
	    			indstart_up[j_up=jseg]=indjseg;
		        } else output[indstart_up[j_up]]=output_up_curr;
	    	} else 
		        output_up_curr=output[i]=input[indstart_up[++j_up]=i];
	        output_low_curr+=(input[i]-output_low_curr)/(i-indstart_low[j_low]+1);      
	        output[indjseg]=output_low_first;
	        while ((j_low>jseg)&&(output_low_curr>=output[ind=indstart_low[j_low-1]]))
	        	output_low_curr+=(output[ind]-output_low_curr)*
		        		((double)(indstart_low[j_low--]-ind)/(i-ind+1));	        		
	        if (j_low==jseg) {
	        	while ((output_low_curr>=output_up_first)&&(jseg<j_up)) {
			    	indjseg2=indstart_up[++jseg];
			    	output_low_curr+=(output_low_curr-output_up_first)*
			    		((double)(indjseg2-indjseg)/(i-indjseg2+1));
			    	while (indjseg<indjseg2) output[indjseg++]=output_up_first;
			    	output_up_first=output[indjseg];
	        	}
	       		if ((indstart_low[j_low=jseg]=indjseg)==i) output_low_first=output_up_first-twolambda;
	       		else output_low_first=output_low_curr; 
	        } else output[indstart_low[j_low]]=output_low_curr;
	    } else {
	        output_up_curr+=((output_low_curr=output[i]=input[indstart_low[++j_low] = i])-
	        	output_up_curr)/(i-indstart_up[j_up]+1);
	        output[indjseg]=output_up_first;
	        while ((j_up>jseg)&&(output_up_curr<=output[ind=indstart_up[j_up-1]]))
	        	output_up_curr+=(output[ind]-output_up_curr)*
		        		((double)(indstart_up[j_up--]-ind)/(i-ind+1));
	        if (j_up==jseg) {
	        	while ((output_up_curr<=output_low_first)&&(jseg<j_low)) {
			    	indjseg2=indstart_low[++jseg];
			    	output_up_curr+=(output_up_curr-output_low_first)*
			    		((double)(indjseg2-indjseg)/(i-indjseg2+1));
			    	while (indjseg<indjseg2) output[indjseg++]=output_low_first;
			    	output_low_first=output[indjseg];
	        	}
    			if ((indstart_up[j_up=jseg]=indjseg)==i) output_up_first=output_low_first+twolambda;
    			else output_up_first=output_up_curr;
	        } else output[indstart_up[j_up]]=output_up_curr;
	    }
	}
	/* here i==width (with value the actual width minus one) */
	if (input[i]+lambda<=output_low_curr) {
        while (jseg<j_low) {
	    	indjseg2=indstart_low[++jseg];
	    	while (indjseg<indjseg2) output[indjseg++]=output_low_first;
	    	output_low_first=output[indjseg];
		}
		while (indjseg<i) output[indjseg++]=output_low_first;
     	output[indjseg]=input[i]+lambda;
	} else if (input[i]-lambda>=output_up_curr) {
		while (jseg<j_up) {
	    	indjseg2=indstart_up[++jseg];
	    	while (indjseg<indjseg2) output[indjseg++]=output_up_first;
	    	output_up_first=output[indjseg];
		}
		while (indjseg<i) output[indjseg++]=output_up_first;
     	output[indjseg]=input[i]-lambda;
	} else {
        output_low_curr+=(input[i]+lambda-output_low_curr)/(i-indstart_low[j_low]+1);      
        output[indjseg]=output_low_first;
        while ((j_low>jseg)&&(output_low_curr>=output[ind=indstart_low[j_low-1]]))
        	output_low_curr+=(output[ind]-output_low_curr)*
		        		((double)(indstart_low[j_low--]-ind)/(i-ind+1));	        		
        if (j_low==jseg) {
        	if (output_up_first>=output_low_curr)
        		while (indjseg<=i) output[indjseg++]=output_low_curr;
        	else {
        		output_up_curr+=(input[i]-lambda-output_up_curr)/(i-indstart_up[j_up]+1);
	        	output[indjseg]=output_up_first;
	        	while ((j_up>jseg)&&(output_up_curr<=output[ind=indstart_up[j_up-1]]))
	        		output_up_curr+=(output[ind]-output_up_curr)*
		        		((double)(indstart_up[j_up--]-ind)/(i-ind+1));
	        	while (jseg<j_up) {
		    		indjseg2=indstart_up[++jseg];
		    		while (indjseg<indjseg2) output[indjseg++]=output_up_first;
		    		output_up_first=output[indjseg];
        		}
        		indjseg=indstart_up[j_up];
        		while (indjseg<=i) output[indjseg++]=output_up_curr;
        	}
        } else {
        	while (jseg<j_low) {
		    	indjseg2=indstart_low[++jseg];
		    	while (indjseg<indjseg2) output[indjseg++]=output_low_first;
		    	output_low_first=output[indjseg];
        	}
        	indjseg=indstart_low[j_low];
        	while (indjseg<=i) output[indjseg++]=output_low_curr;
        }
	}
	free(indstart_low);
	free(indstart_up);
}
