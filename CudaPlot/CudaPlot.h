#ifndef __CUDA_PLOT_H__
#define __CUDA_PLOT_H__

#include "vector_types.h"

#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <string.h>

#include "CudaUtility.h"

#define MAX_TRACES 4

class CudaChartArray
{
public:

	CudaChartArray(int rows, int cols, int chartArrayWidth, int chartArrayHeight, int margin, int padding,
		int aggregateWidth, int aggregateHeight,
		uchar4 windowBackgroundColor, uchar4 chartBackgroundColor, uchar4 chartSelectedColor,
		uchar4 chartFrameColor, uchar4 chartAxisColor, uchar4 chartPlotColor,
		int2 xRange, int2 yRange, int maxNumDataPoints, int numTraces);

	~CudaChartArray();

	int2 GetChartArrayPixelSize();

	int GetMaxNumberOfTraces();
	void SetTraceVisibility(int traceNum, bool isVisible);
	
	bool AllocateForData(int numDataPoints);
	bool AllocateForSelected();
	bool AllocateForChartImage();
	bool AllocateForAggregateImage();

	void SetSelected();
	void SetTraceColor(int traceNum, uchar4 color);
	void CalcConversionFactors();

	void AppendData(int2 *p_new_points, int traceNum);

	void Redraw();
	void AppendLine(); // just draw connect the last two points

	uchar4* GetChartImagePtr();

	void SetWindowBackground(uchar4 color);
	void SetInitialRanges(int xmin, int xmax, int ymin, int ymax);
	
	void RedrawAggregate();
	void AppendLineAggregate();
	uchar4* GetAggregateImagePtr();
	void Resize(int chartArrayWidth, int chartArrayHeight, int aggregateWidth, int aggregateHeight);

	bool* GetSelectionArrayPtr();

	int32_t GetChartWidth() { return m_chart_width; }
	int32_t GetChartHeight() { return m_chart_height; }

	int32_t GetRowFromY(int32_t y);
	int32_t GetColumnFromX(int32_t x);

	void Reset();


	int m_threadsPerBlock;

	// number of rows and columns in chart array
	int m_rows;
	int m_cols;

	//  colors used 	
	uchar4 m_window_background_color;
	uchar4 m_chart_background_color;
	uchar4 m_chart_selected_color;
	uchar4 m_chart_frame_color;
	uchar4 m_chart_axis_color;

	uchar4 m_trace_color[MAX_TRACES];

	// pixel size of window that contains this array of charts
	int32_t m_chartArray_width;
	int32_t m_chartArray_height;


	// pixel size of each chart panel (this is calculated from the window size and the number of rows/cols)
	int32_t m_chart_width;
	int32_t m_chart_height;

	int32_t m_margin;
	int32_t m_padding;

	// pointers to data block where the plotted data is stored
	int2* mp_d_data[MAX_TRACES];  // device pointer to data block
	int m_numTraces;  // number of traces that have been set up
	bool m_traceVisible[MAX_TRACES];  // flag array indicating which traces are visible
	

    // pointer to the image of the chart array
	uchar4 *mp_h_chart_image; // host pointer 
	uchar4 *mp_d_chart_image; // device pointer
	size_t m_chart_image_pitch;

	uchar4 *mp_h_aggregate_image; // host pointer	
	uchar4 *mp_d_aggregate_image; // device pointer
	size_t m_aggregate_image_pitch;

	// pixel size of window that contains this aggregate chart
	int32_t m_aggregate_width;
	int32_t m_aggregate_height;


	// selected flag array - this a pointer to an array of bools indicates whether a particular chart is selected or not
	bool  *mp_d_chart_selected; // device pointer
	bool  *mp_h_chart_selected; // host pointer


	int32_t  m_num_data_points[MAX_TRACES];  // number of data points currently in memory
	int32_t  m_max_num_data_points;  // max number of data points possible with current memory allocation.
									 // if this number is reached, a larger memory block can be allocated.


	// chart range for each axis (this is the set range of values that are to be plotted)
	int32_t  m_x_min;
	int32_t  m_x_max;
	int32_t  m_y_min;
	int32_t  m_y_max;

	// value-to-pixel conversion factors (this factor allows for the calculation of a pixel location of a point from it's value)
	//  these will are set (or will change) whenever:
	//		1 - a new windows size is set
	//		2 - a new chart range is set
	float  m_x_value_to_pixel;
	float  m_y_value_to_pixel;
	float  m_x_value_to_pixel_aggregate;
	float  m_y_value_to_pixel_aggregate;

	int2   m_initial_xRange;
	int2   m_initial_yRange;
};




#endif