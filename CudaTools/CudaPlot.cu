#include "CudaPlot.h"

#define LOG_CUDA

// CONSTRUCTOR
CudaChartArray::CudaChartArray(int rows, int cols, 
	int chartArrayWidth, int chartArrayHeight,
	int margin, int padding,
	int aggregateWidth, int aggregateHeight,
	uchar4 windowBackgroundColor, uchar4 chartBackgroundColor, uchar4 chartSelectedColor,
	uchar4 chartFrameColor, uchar4 chartAxisColor, uchar4 chartPlotColor,
	int2 xRange, int2 yRange, int maxNumDataPoints, int numTraces)
{
	m_threadsPerBlock = 256;  // how many threads allocated for drawing each chart
	if (numTraces > MAX_TRACES) numTraces = MAX_TRACES;

	m_rows = rows;
	m_cols = cols;
	m_chartArray_width = chartArrayWidth;
	m_chartArray_height = chartArrayHeight;
	m_numTraces = numTraces;

	m_aggregate_width = aggregateWidth;
	m_aggregate_height = aggregateHeight;

	m_margin = margin;
	m_padding = padding;

	m_window_background_color = windowBackgroundColor;
	m_chart_background_color = chartBackgroundColor;
	m_chart_selected_color = chartSelectedColor;
	m_chart_frame_color = chartFrameColor;
	m_chart_axis_color = chartAxisColor;

	// set default colors
	for(int i = 0; i<m_numTraces; i++)
		m_trace_color[i] = make_uchar4(0, 255, 255, 255);

	m_chart_width = (m_chartArray_width - (2 * m_margin) - ((m_cols - 1)*m_padding)) / m_cols;
	m_chart_height = (m_chartArray_height - (2 * m_margin) - ((m_rows - 1)*m_padding)) / m_rows;

	m_x_min = xRange.x;
	m_x_max = xRange.y;
	m_y_min = yRange.x;
	m_y_max = yRange.y;

	m_initial_xRange.x = m_x_min;
	m_initial_xRange.y = m_x_max;
	m_initial_yRange.x = m_y_min;
	m_initial_yRange.y = m_y_max;

	m_max_num_data_points = maxNumDataPoints;
	for(int i = 0; i<m_numTraces; i++)
		m_num_data_points[i] = 0;

	bool success;
	success = AllocateForData(maxNumDataPoints);
	success = AllocateForChartImage();
	success = AllocateForAggregateImage();
	success = AllocateForSelected();

	cudaError_t result = cudaMallocHost((uchar4**)&mp_h_chart_image, m_chartArray_width * m_chartArray_height * sizeof(uchar4));
	result = cudaMallocHost((uchar4**)&mp_h_aggregate_image, m_aggregate_width * m_aggregate_height * sizeof(uchar4));

	result = cudaMallocHost((bool**)&mp_h_chart_selected, m_rows * m_cols * sizeof(bool));

	CalcConversionFactors();

}


CudaChartArray::~CudaChartArray()
{	
	for (int i = 0; i < m_numTraces; i++)
	{
		cudaFree(mp_d_data[i]);
	}

	cudaFree(mp_d_chart_selected);
	cudaFree(mp_d_chart_image);
	cudaFree(mp_d_aggregate_image);

	cudaFree(mp_h_chart_image);
	cudaFree(mp_h_aggregate_image);
	cudaFree(mp_h_chart_selected);
}


//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

//   Cuda Kernels


__global__ void drawEmptyCharts(uchar4* output, uint32_t windowWidth, uint32_t windowHeight, size_t windowPitch,
	uint32_t numRows, uint32_t numCols,
	uint32_t margin, uint32_t padding,
	uint32_t chartWidth, uint32_t chartHeight,
	int32_t yMin, int32_t yMax,
	uchar4 windowBackgroundColor, uchar4 chartBackgroundColor, uchar4 selectedBackgroundColor,
	uchar4 frameColor, uchar4 axisColor,
	bool* selected)
{
	// calc x,y position of pixel to operate on
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x; // column of pixel inside panel
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y; // row of pixel inside panel

														// make sure we don't try to operate outside the image
	if (x >= windowWidth) return;
	if (y >= windowHeight) return;

	uchar4 color = windowBackgroundColor;

	// find pixel offset of x axis from bottom of chart (used to draw the x axis)
	int axisOffset = (int)((float)(0 - yMin) / (float)(yMax - yMin) * (float)(chartHeight));


	int foundRow = -1;
	int foundCol = -1;
	bool insideRow = false;
	bool insideCol = false;
	bool onAxis = false;

	for (int r = 0; r < numRows; r++)
	{
		int ndx = r*numCols;
		int y1 = margin + (r * (chartHeight + padding));
		int y2 = y1 + chartHeight - 1;

		if (y >= y1 && y <= y2)
		{
			insideRow = true;
			foundRow = r;
			if (y == (y2 - axisOffset)) // pixel is on x axis of chart					
				onAxis = true;
			break;
		}
	}
	for (int c = 0; c < numCols; c++)
	{		
		int x1 = margin + (c * (chartWidth + padding));
		int x2 = x1 + chartWidth - 1;

		if (x >= x1 && x <= x2)
		{
			insideCol = true;
			foundCol = c;
			break;
		}
	}

	if (insideRow && insideCol)
	{		
		if (onAxis)
			color = axisColor;
		else if (selected[foundRow * numCols + foundCol])
			color = selectedBackgroundColor;
		else
			color = chartBackgroundColor;
	}

	// calculate pixel position in array
	uint8_t* byteArray = (uint8_t*)&output[0];
	uchar4*  pixelPtr = (uchar4*)&byteArray[(y * windowPitch) + (x * sizeof(uchar4))];

	*pixelPtr = color;
}


__global__ void plotChart(int2* data, uchar4* output, int numPoints, int maxNumPoints, float convX, float convY,
	int panelWidth, int panelHeight, size_t windowPitch,
	int plotMinX, int plotMaxX, int plotMinY, int plotMaxY,
	uchar4 plotColor, int margin, int padding)
{
	// 1 thread per chart, so blockDim should be (totalChartCols, totalChartRows)
	// get the chart array row/col that this chart belongs to
	int chartRow = blockIdx.y;
	int chartCol = blockIdx.x;
	int numRows = gridDim.y;
	int numCols = gridDim.x;
	int threadNum = threadIdx.x; // this is the thread number for threads inside this block

	if (numPoints < 1) return;

	int pointsPerThread = (numPoints / blockDim.x) + 2;

	
	// get pixel coordinates within the entire plot array window of chart origin (lower left corner)
	int ndx0 = chartRow * numCols + chartCol;

	// x1 - x minimum for chart(r,c)
	uint32_t chartOriginX = margin + (chartCol * (panelWidth + padding));
	// y2 - y maximum for chart(r,c)
	uint32_t chartOriginY = margin + (chartRow * (panelHeight + padding)) + panelHeight - 1;


	// get pointer to data for this chart	
	uint32_t pixelNdx;

	int start = threadNum * pointsPerThread;
	int end = start + pointsPerThread - 1;

	/*if (numPoints > 100)
	{
		threadNum = threadIdx.x;
	}*/

	if (start >= numPoints) return;
	if (end >= numPoints) end = numPoints - 1;

	int dataPitch = numRows * numCols;
	int chartNum = (chartRow * numCols) + chartCol;

	for (int j = start; j < end; j++)
	{
		// Bresenham line drawing algorithm
		// NOTE: this conversion assumes that pixels start at 0
		int ndx = (j*dataPitch) + chartNum;
		int2 pt1 = data[ndx];
		int2 pt2 = data[ndx + dataPitch];

		if (chartNum == 5 && j == 10)
		{
			int i = 0;
			chartNum += i;
		}

		int x1 = (int)((float)(pt1.x - plotMinX) * convX); // convert to pixel coordinates
		int y1 = (int)((float)(pt1.y - plotMinY) * convY);
		int x2 = (int)((float)(pt2.x - plotMinX) * convX);
		int y2 = (int)((float)(pt2.y - plotMinY) * convY);

		int x, y, dx, dy, dx1, dy1, px, py, xe, ye, i;
		dx = x2 - x1;
		dy = y2 - y1;
		dx1 = fabs((double)dx);
		dy1 = fabs((double)dy);
		px = 2 * dy1 - dx1;
		py = 2 * dx1 - dy1;
		if (dy1 <= dx1)
		{
			if (dx >= 0)
			{
				x = x1;
				y = y1;
				xe = x2;
			}
			else
			{
				x = x2;
				y = y2;
				xe = x1;
			}


			// draw pixel
			if (x < panelWidth && y < panelHeight)
			{
				//pixelNdx = ((chartOriginY - y) * (windowPitch)) + (chartOriginX + x);
				//output[pixelNdx] = plotColor;

				// calculate pixel position in array
				//uint8_t* byteArray = (uint8_t*)&output[0];
				//uchar4*  pixelPtr = (uchar4*)&byteArray[((chartOriginY - y) * windowPitch) + ((chartOriginX + x) * sizeof(uchar4))];
				//*pixelPtr = plotColor;

				unsigned char* ptr = (unsigned char*)&output[(chartOriginY - y) * (windowPitch / sizeof(uchar4)) + (chartOriginX + x)];

				output[(chartOriginY - y) * (windowPitch/sizeof(uchar4)) + (chartOriginX + x)] = plotColor;

			}

			for (i = 0; x<xe; i++)
			{
				x = x + 1;
				if (px<0)
				{
					px = px + 2 * dy1;
				}
				else
				{
					if ((dx<0 && dy<0) || (dx>0 && dy>0))
					{
						y = y + 1;
					}
					else
					{
						y = y - 1;
					}
					px = px + 2 * (dy1 - dx1);
				}


				// draw pixel
				if (x < panelWidth && y < panelHeight)
				{
					//pixelNdx = ((chartOriginY - y) * (windowPitch)) + (chartOriginX + x);
					//output[pixelNdx] = plotColor;

					// calculate pixel position in array
					//uint8_t* byteArray = (uint8_t*)&output[0];
					//uchar4*  pixelPtr = (uchar4*)&byteArray[((chartOriginY - y) * windowPitch) + ((chartOriginX + x) * sizeof(uchar4))];
					//*pixelPtr = plotColor;

					unsigned char* ptr = (unsigned char*)&output[(chartOriginY - y) * (windowPitch / sizeof(uchar4)) + (chartOriginX + x)];

					output[(chartOriginY - y) * (windowPitch / sizeof(uchar4)) + (chartOriginX + x)] = plotColor;
				}
			}
		}
		else
		{
			if (dy >= 0)
			{
				x = x1;
				y = y1;
				ye = y2;
			}
			else
			{
				x = x2;
				y = y2;
				ye = y1;
			}

			// draw pixel
			if (x < panelWidth && y < panelHeight)
			{
				//pixelNdx = ((chartOriginY - y) * windowPitch) + (chartOriginX + x);
				//output[pixelNdx] = plotColor;

				// calculate pixel position in array
				//uint8_t* byteArray = (uint8_t*)&output[0];
				//uchar4*  pixelPtr = (uchar4*)&byteArray[((chartOriginY - y) * windowPitch) + ((chartOriginX + x) * sizeof(uchar4))];
				//*pixelPtr = plotColor;

				unsigned char* ptr = (unsigned char*)&output[(chartOriginY - y) * (windowPitch / sizeof(uchar4)) + (chartOriginX + x)];

				output[(chartOriginY - y) * (windowPitch / sizeof(uchar4)) + (chartOriginX + x)] = plotColor;
			}

			for (i = 0; y<ye; i++)
			{
				y = y + 1;
				if (py <= 0)
				{
					py = py + 2 * dx1;
				}
				else
				{
					if ((dx<0 && dy<0) || (dx>0 && dy>0))
					{
						x = x + 1;
					}
					else
					{
						x = x - 1;
					}
					py = py + 2 * (dx1 - dy1);
				}

				// draw pixel
				if (x < panelWidth && y < panelHeight)
				{
					//pixelNdx = ((chartOriginY - y) * windowPitch) + (chartOriginX + x);
					//output[pixelNdx] = plotColor;

					// calculate pixel position in array
					//uint8_t* byteArray = (uint8_t*)&output[0];
					//uchar4*  pixelPtr = (uchar4*)&byteArray[((chartOriginY - y) * windowPitch) + ((chartOriginX + x) * sizeof(uchar4))];
					//*pixelPtr = plotColor;

					unsigned char* ptr = (unsigned char*)&output[(chartOriginY - y) * (windowPitch / sizeof(uchar4)) + (chartOriginX + x)];

					output[(chartOriginY - y) * (windowPitch / sizeof(uchar4)) + (chartOriginX + x)] = plotColor;
				}
			}
		}
	}

}








__global__ void drawEmptyAggregateChart(uchar4* output, uint32_t windowWidth, uint32_t windowHeight, size_t windowPitch,
	uint32_t numRows, uint32_t numCols,
	uint32_t margin,
	int32_t yMin, int32_t yMax,
	uchar4 windowBackgroundColor, uchar4 chartBackgroundColor, 
	uchar4 frameColor, uchar4 axisColor)
{
	// calc x,y position of pixel to operate on
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x; // column of pixel inside panel
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y; // row of pixel inside panel

														// make sure we don't try to operate outside the image
	if (x >= windowWidth) return;
	if (y >= windowHeight) return;

	uchar4 color = windowBackgroundColor;
	

	// find pixel offset of x axis from bottom of chart (used to draw the x axis)
	int axisOffset = (int)((float)(0 - yMin) / (float)(yMax - yMin) * (float)(windowHeight));

	if (y == (windowHeight - axisOffset)) // pixel is on x axis of chart					
		color = axisColor;
	else // pixel inside unselected chart
		color = chartBackgroundColor;

	// calculate pixel position in array
	uint8_t* byteArray = (uint8_t*)&output[0];
	uchar4*  pixelPtr = (uchar4*)&byteArray[(y * windowPitch) + (x * sizeof(uchar4))];

	*pixelPtr = color;

}






__global__ void plotAggregateChart(int2* data, uchar4* output, int numPoints, int maxNumPoints, float convX, float convY,
	int windowWidth, int windowHeight, size_t windowPitch,
	int plotMinX, int plotMaxX, int plotMinY, int plotMaxY,
	uchar4 plotColor, int margin, bool* selected)
{
	// dim3 threadsPerBlock1(m_threadsPerBlock, 1);  // block dims
	// dim3 numBlocks1(m_cols, m_rows);  // grid dims
	int chartRow = blockIdx.y;
	int chartCol = blockIdx.x;
	int numRows = gridDim.y;
	int numCols = gridDim.x;
	int threadNum = threadIdx.x; // this is the thread number for threads inside this block
	int numThreads = blockDim.x;

	if (numPoints < 1) return;

	// Is this chart selected?  If so, then it's data should be plotted.  If not, just return
	bool chartSelected = selected[(chartRow * numCols) + chartCol];
	if (!chartSelected) return;


	int pointsPerThread = (numPoints / numThreads) + 2;

	// get pixel coordinates within the entire plot array window of chart origin (lower left corner)	
	uint32_t chartOriginX = margin;
	uint32_t chartOriginY = windowHeight - margin;

	// get pointer to data for this chart	
	uint32_t pixelNdx;

	int start = threadNum * pointsPerThread;
	int end = start + pointsPerThread;

	if (start >= numPoints) return;
	if (end >= numPoints) end = numPoints - 1;

	// each group of data points make up one row of data
	//  int dataRowPitch = numChartRows * numChartCols;
	//  So, to get the data point (pt) for row = r, col = c, and data point number = n
	//	int2 pt = mp_d_data[(dataRowLength * n) + (r * m_cols) + c];

	int dataPitch = numRows * numCols;
	int chartNum = (chartRow * numCols) + chartCol;

	for (int j = start; j < end; j++)
	{
		// Bresenham line drawing algorithm
		// NOTE: this conversion assumes that pixels start at 0
		int ndx = (j*dataPitch) + chartNum;
		int2 pt1 = data[ndx];
		int2 pt2 = data[ndx + dataPitch];

		if (chartNum == 5 && j == 10)
		{
			int i = 0;
			chartNum += i;
		}

		int x1 = (int)((float)(pt1.x - plotMinX) * convX); // convert to pixel coordinates
		int y1 = (int)((float)(pt1.y - plotMinY) * convY);
		int x2 = (int)((float)(pt2.x - plotMinX) * convX);
		int y2 = (int)((float)(pt2.y - plotMinY) * convY);

		int x, y, dx, dy, dx1, dy1, px, py, xe, ye, i;
		dx = x2 - x1;
		dy = y2 - y1;
		dx1 = fabs((double)dx);
		dy1 = fabs((double)dy);
		px = 2 * dy1 - dx1;
		py = 2 * dx1 - dy1;
		if (dy1 <= dx1)
		{
			if (dx >= 0)
			{
				x = x1;
				y = y1;
				xe = x2;
			}
			else
			{
				x = x2;
				y = y2;
				xe = x1;
			}


			// draw pixel
			if (x < windowWidth && y < windowHeight)
			{
				//pixelNdx = ((chartOriginY - y) * (windowPitch)) + (chartOriginX + x);
				//output[pixelNdx] = plotColor;

				// calculate pixel position in array
				//uint8_t* byteArray = (uint8_t*)&output[0];
				//uchar4*  pixelPtr = (uchar4*)&byteArray[((chartOriginY - y) * windowPitch) + ((chartOriginX + x) * sizeof(uchar4))];
				//*pixelPtr = plotColor;

				unsigned char* ptr = (unsigned char*)&output[(chartOriginY - y) * (windowPitch / sizeof(uchar4)) + (chartOriginX + x)];

				output[(chartOriginY - y) * (windowPitch / sizeof(uchar4)) + (chartOriginX + x)] = plotColor;
			}

			for (i = 0; x<xe; i++)
			{
				x = x + 1;
				if (px<0)
				{
					px = px + 2 * dy1;
				}
				else
				{
					if ((dx<0 && dy<0) || (dx>0 && dy>0))
					{
						y = y + 1;
					}
					else
					{
						y = y - 1;
					}
					px = px + 2 * (dy1 - dx1);
				}


				// draw pixel
				if (x < windowWidth && y < windowHeight)
				{
					//pixelNdx = ((chartOriginY - y) * (windowPitch)) + (chartOriginX + x);
					//output[pixelNdx] = plotColor;

					// calculate pixel position in array
					//uint8_t* byteArray = (uint8_t*)&output[0];
					//uchar4*  pixelPtr = (uchar4*)&byteArray[((chartOriginY - y) * windowPitch) + ((chartOriginX + x) * sizeof(uchar4))];
					//*pixelPtr = plotColor;

					unsigned char* ptr = (unsigned char*)&output[(chartOriginY - y) * (windowPitch / sizeof(uchar4)) + (chartOriginX + x)];

					output[(chartOriginY - y) * (windowPitch / sizeof(uchar4)) + (chartOriginX + x)] = plotColor;
				}
			}
		}
		else
		{
			if (dy >= 0)
			{
				x = x1;
				y = y1;
				ye = y2;
			}
			else
			{
				x = x2;
				y = y2;
				ye = y1;
			}

			// draw pixel
			if (x < windowWidth && y < windowHeight)
			{
				//pixelNdx = ((chartOriginY - y) * windowPitch) + (chartOriginX + x);
				//output[pixelNdx] = plotColor;

				// calculate pixel position in array
				//uint8_t* byteArray = (uint8_t*)&output[0];
				//uchar4*  pixelPtr = (uchar4*)&byteArray[((chartOriginY - y) * windowPitch) + ((chartOriginX + x) * sizeof(uchar4))];
				//*pixelPtr = plotColor;

				unsigned char* ptr = (unsigned char*)&output[(chartOriginY - y) * (windowPitch / sizeof(uchar4)) + (chartOriginX + x)];

				output[(chartOriginY - y) * (windowPitch / sizeof(uchar4)) + (chartOriginX + x)] = plotColor;
			}

			for (i = 0; y<ye; i++)
			{
				y = y + 1;
				if (py <= 0)
				{
					py = py + 2 * dx1;
				}
				else
				{
					if ((dx<0 && dy<0) || (dx>0 && dy>0))
					{
						x = x + 1;
					}
					else
					{
						x = x - 1;
					}
					py = py + 2 * (dx1 - dy1);
				}

				// draw pixel
				if (x < windowWidth && y < windowHeight)
				{
					//pixelNdx = ((chartOriginY - y) * windowPitch) + (chartOriginX + x);
					//output[pixelNdx] = plotColor;

					// calculate pixel position in array
					//uint8_t* byteArray = (uint8_t*)&output[0];
					//uchar4*  pixelPtr = (uchar4*)&byteArray[((chartOriginY - y) * windowPitch) + ((chartOriginX + x) * sizeof(uchar4))];
					//*pixelPtr = plotColor;

					unsigned char* ptr = (unsigned char*)&output[(chartOriginY - y) * (windowPitch / sizeof(uchar4)) + (chartOriginX + x)];

					output[(chartOriginY - y) * (windowPitch / sizeof(uchar4)) + (chartOriginX + x)] = plotColor;
				}
			}
		}
	}

}




// END Cuda Kernels

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////


int CudaChartArray::GetMaxNumberOfTraces()
{
	return MAX_TRACES;
}

void CudaChartArray::SetTraceVisibility(int traceNum, bool isVisible)
{
	if (traceNum > m_numTraces - 1 || traceNum < 0) return;
	m_traceVisible[traceNum] = isVisible;
}


int2 CudaChartArray::GetChartArrayPixelSize()
{
	return make_int2(m_chartArray_width, m_chartArray_height);
}


bool CudaChartArray::AllocateForData(int numDataPoints)

{
	// each group of data points make up one row of data
	//
	//  int dataRowLength = numChartRows * numChartCols * sizeof(int2);
	//
	// So, to get the data point (pt) for row = r, col = c, and data point number = n
	//
	//	int2 pt = mp_d_data[(dataRowLength * n) + (r * m_cols) + c];


	size_t blocksize = m_rows * m_cols * sizeof(int2);  // this is the size of one group of data (one point for each chart)

	m_max_num_data_points = numDataPoints;  // this is the max number of blocks that can be added (max points for each chart)

	size_t size = blocksize * m_max_num_data_points; // size of memory allocated for data points

	bool success = true;

	for (int i = 0; i < m_numTraces; i++)
	{
		cudaError_t result = cudaMalloc((void**)&mp_d_data[i], size);
		if (result != CUDA_SUCCESS)
		{
			printf(LOG_CUDA "failed to allocated %zu bytes for data block\n", size);
			success = false;
			m_traceVisible[i] = false;
			break;
		}
		m_traceVisible[i] = true;
	}

	return success;
}

bool CudaChartArray::AllocateForSelected()
{
	// this allocates a memory buffer to hold the boolean array, each position containing a
	// flag indicating whether a chart is selected (so is drawn with a background that is a 
	// different color).
	//
	//	In order to retrieve the selected flag for the chart at r,c:
	//
	//	bool b = mp_d_chart_selected[(r * m_cols) + c];

	size_t size = m_rows * m_cols * sizeof(bool);
	bool success = true;

	cudaError_t result = cudaMalloc((void**)&mp_d_chart_selected, size);
	if (result != CUDA_SUCCESS)
	{
		printf(LOG_CUDA "failed to allocated %zu bytes for selected array\n", size);
		success = false;
	}

	return success;
}

bool CudaChartArray::AllocateForChartImage()
{
	size_t size = m_chartArray_width * m_chartArray_height * sizeof(uchar4);
	bool success = true;

	cudaError_t result = cudaMallocPitch(&mp_d_chart_image, &m_chart_image_pitch, m_chartArray_width * sizeof(uchar4), m_chartArray_height);
	if (result != CUDA_SUCCESS)
	{
		printf(LOG_CUDA "failed to allocated %zu bytes for chart image\n", size);
		success = false;
	}

	return success;
}


bool CudaChartArray::AllocateForAggregateImage()
{
	size_t size = m_aggregate_width * m_aggregate_height * sizeof(uchar4);
	bool success = true;

	cudaError_t result = cudaMallocPitch(&mp_d_aggregate_image, &m_aggregate_image_pitch, m_aggregate_width * sizeof(uchar4), m_aggregate_height);
	if (result != CUDA_SUCCESS)
	{
		printf(LOG_CUDA "failed to allocated %zu bytes for aggregate image\n", size);
		success = false;
	}

	return success;
}


void CudaChartArray::SetSelected()
{
	size_t size = m_rows * m_cols * sizeof(bool);
	cudaError_t err = cudaMemcpy((bool*)mp_d_chart_selected, (bool*)mp_h_chart_selected, m_rows * m_cols * sizeof(bool), cudaMemcpyHostToDevice);
	if (err != CUDA_SUCCESS)
	{
		printf(LOG_CUDA "failed to copy selected chart array to device\n");
	}
	else
	{
		Redraw();
	}
}

void CudaChartArray::SetTraceColor(int traceNum, uchar4 color)
{
	if (traceNum > MAX_TRACES-1 || traceNum < 0) return;
	m_trace_color[traceNum] = color;
}

void CudaChartArray::CalcConversionFactors()
{
	// conversion factor definition
	//
	//                   [v----conversion factor----v]
	//	px = (x - xmin)( (pxmax - pxmin)/(xmax - xmin) ) + pxmin

	// the equation above converts a axis value to a pixel location.
	//		x	= value to be plotted
	//		px	= pixel location for value x
	//		xmin,xmax = value range
	//		pxmin, pxmax = pixel coordinate range (typically comes from the size of the chart panel, and pxmin is usually 0)

	m_x_value_to_pixel = (float)(m_chart_width) / (float)(m_x_max - m_x_min);
	m_y_value_to_pixel = (float)(m_chart_height) / (float)(m_y_max - m_y_min);

	m_x_value_to_pixel_aggregate = (float)(m_aggregate_width - (2 * m_margin)) / (float)(m_x_max - m_x_min);
	m_y_value_to_pixel_aggregate = (float)(m_aggregate_height - (2 * m_margin)) / (float)(m_y_max - m_y_min);

}

void CudaChartArray::AppendData(int2 *p_new_points, int traceNum)
{
	bool rangeChanged = false;

	if (traceNum > m_numTraces - 1 || traceNum < 0) return;

	uint8_t* ptr = (uint8_t*)mp_d_data[traceNum] + (m_num_data_points[traceNum] * m_rows * m_cols * sizeof(int2));


	cudaError_t result = cudaMemcpy(ptr, (void*)p_new_points, m_rows * m_cols * sizeof(int2), cudaMemcpyHostToDevice);

	if (result == CUDA_SUCCESS)
	{
		for (int i = 0; i < m_rows * m_cols; i++)
		{
			if (p_new_points[i].x > m_x_max) { m_x_max = p_new_points[i].x * 11 / 10; rangeChanged = true; }
			if (p_new_points[i].y > m_y_max) { m_y_max = p_new_points[i].y * 6 / 5;	rangeChanged = true; }
			if (p_new_points[i].x < m_x_min) { m_x_min = p_new_points[i].x;	rangeChanged = true; }
			if (p_new_points[i].y < m_y_min) { m_y_min = p_new_points[i].y; rangeChanged = true; }
		}

		m_num_data_points[traceNum]++;

		if (rangeChanged)
		{
			CalcConversionFactors();
		}
	}
	else
	{
		printf(LOG_CUDA "failed to copy new data points to GPU.  %zu bytes for selected array\n", m_rows*m_cols * sizeof(int2));
	}

}



void CudaChartArray::Redraw()
{
	dim3 threadsPerBlock(32, 16);
	dim3 numBlocks;
	numBlocks.x = (m_chartArray_width + threadsPerBlock.x - 1) / threadsPerBlock.x;
	numBlocks.y = (m_chartArray_height + threadsPerBlock.y - 1) / threadsPerBlock.y;

	drawEmptyCharts << <numBlocks, threadsPerBlock >> > (mp_d_chart_image, m_chartArray_width, m_chartArray_height, m_chart_image_pitch,
		m_rows, m_cols, m_margin, m_padding, m_chart_width, m_chart_height, m_y_min, m_y_max,
		m_window_background_color, m_chart_background_color, m_chart_selected_color, m_chart_frame_color, m_chart_axis_color,
		mp_d_chart_selected);

	//  create 1 thread for each chart
	dim3 threadsPerBlock1(m_threadsPerBlock, 1);  // block dims
	dim3 numBlocks1(m_cols, m_rows);  // grid dims

	// redraw chart
	for (int i = 0; i < m_numTraces; i++)
	{
		if(m_traceVisible[i])
			plotChart << <numBlocks1, threadsPerBlock1 >> > (mp_d_data[i], mp_d_chart_image, m_num_data_points[i], 
				m_max_num_data_points, m_x_value_to_pixel, m_y_value_to_pixel,
				m_chart_width, m_chart_height, m_chart_image_pitch,
				m_x_min, m_x_max, m_y_min, m_y_max,
				m_trace_color[i], m_margin, m_padding);
	}

	// copy full chart array image to host
	cudaError_t result = cudaMemcpy2D(mp_h_chart_image, m_cols * sizeof(uchar4), mp_d_chart_image, m_chart_image_pitch, m_cols * sizeof(uchar4), m_rows, 
		cudaMemcpyDeviceToHost);
	if (result != CUDA_SUCCESS)
	{
		printf(LOG_CUDA "failed to copy new image from GPU.");
	}
}


void CudaChartArray::AppendLine()
{
	// setup arguments
	//  create 1 thread for each chart
	const dim3 block(m_threadsPerBlock, 1);
	const dim3 grid(m_cols, m_rows);
	//const dim3 block(m_cols, m_rows);
	//const dim3 grid(1, 1);

	// a block threads per character to render
	// a grid containing one block for each letter

	for (int i = 0; i < m_numTraces; i++)
	{
		if (m_traceVisible[i])
			plotChart << <grid, block >> > (mp_d_data[i], mp_d_chart_image, m_num_data_points[i], 
				m_max_num_data_points, m_x_value_to_pixel, m_y_value_to_pixel,
				m_chart_width, m_chart_height, m_chart_image_pitch,
				m_x_min, m_x_max, m_y_min, m_y_max,
				m_trace_color[i], m_margin, m_padding);
	}

	// copy full chart array image to host
	cudaMemcpy2D(mp_h_chart_image, m_cols * sizeof(uchar4), mp_d_chart_image, m_chart_image_pitch, m_cols * sizeof(uchar4), m_rows, 
		cudaMemcpyDeviceToHost);

}

uchar4 * CudaChartArray::GetChartImagePtr()
{	
	cudaError_t result = cudaMemcpy2D(mp_h_chart_image, m_chartArray_width * sizeof(uchar4), 
									  mp_d_chart_image, m_chart_image_pitch, 
									  m_chartArray_width * sizeof(uchar4), m_chartArray_height, cudaMemcpyDeviceToHost);

	return mp_h_chart_image; 
}

void CudaChartArray::SetWindowBackground(uchar4 color)
{
	m_window_background_color = color;
}

void CudaChartArray::SetInitialRanges(int xmin, int xmax, int ymin, int ymax)
{
	m_initial_xRange.x = xmin;
	m_initial_xRange.y = xmax;
	m_initial_yRange.x = ymin;
	m_initial_yRange.y = ymax;
}



int32_t CudaChartArray::GetRowFromY(int32_t y)
{
	int32_t row = -1;

	for (int r = 0; r < m_rows; r++)
	{
		int ndx = r * m_cols;
		// .x = x1, .y = y1, .w = x2, .z = y2
		int y1 = m_margin + (r * (m_chart_height + m_padding));
		int y2 = y1 + m_chart_height - 1;
		if (y >= y1 && y <= y2)
		{
			row = r;
			break;
		}
	}

	return row;
}


int32_t CudaChartArray::GetColumnFromX(int32_t x)
{
	int32_t col = -1;

	for (int c = 0; c < m_cols; c++)
	{
		// .x = x1, .y = y1, .w = x2, .z = y2
		int x1 = m_margin + (c * (m_chart_width + m_padding));
		int x2 = x1 + m_chart_width - 1;

		if (x >= x1 && x <= x2)
		{
			col = c;
			break;
		}
	}

	return col;
}





//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////


void CudaChartArray::RedrawAggregate()
{
	dim3 threadsPerBlock(32, 16);
	dim3 numBlocks;
	numBlocks.x = (m_aggregate_width + threadsPerBlock.x - 1) / threadsPerBlock.x;
	numBlocks.y = (m_aggregate_height + threadsPerBlock.y - 1) / threadsPerBlock.y;

	drawEmptyAggregateChart << <numBlocks, threadsPerBlock >> > (mp_d_aggregate_image, m_aggregate_width, m_aggregate_height, m_aggregate_image_pitch,
		m_rows, m_cols, m_margin, m_y_min, m_y_max,
		m_window_background_color, m_chart_background_color, m_chart_frame_color, m_chart_axis_color);

	//  create 1 thread for each chart
	dim3 threadsPerBlock1(m_threadsPerBlock, 1);  // block dims
	dim3 numBlocks1(m_cols, m_rows);  // grid dims
	

	// redraw chart
	for (int i = 0; i < m_numTraces; i++)
	{
		if (m_traceVisible[i])
			plotAggregateChart << <numBlocks1, threadsPerBlock1 >> > (mp_d_data[i], mp_d_aggregate_image, 
				m_num_data_points[i], m_max_num_data_points,
				m_x_value_to_pixel_aggregate, m_y_value_to_pixel_aggregate,
				m_aggregate_width, m_aggregate_height, m_aggregate_image_pitch,
				m_x_min, m_x_max, m_y_min, m_y_max,
				m_trace_color[i], m_margin, mp_d_chart_selected);
	}

	// copy full aggregate image to host
	cudaMemcpy2D(mp_h_aggregate_image, m_cols * sizeof(uchar4), mp_d_aggregate_image, m_aggregate_image_pitch, m_cols * sizeof(uchar4), m_rows, 
		cudaMemcpyDeviceToHost);

}



void CudaChartArray::AppendLineAggregate()
{

	//  create 1 thread for each chart
	dim3 threadsPerBlock1(m_threadsPerBlock, 1);  // block dims
	dim3 numBlocks1(m_cols, m_rows);  // grid dims

	for (int i = 0; i < m_numTraces; i++)
	{
		// redraw chart
		if (m_traceVisible[i])
			plotAggregateChart << <numBlocks1, threadsPerBlock1 >> > (mp_d_data[i], mp_d_aggregate_image, 
				m_num_data_points[i], m_max_num_data_points,
				m_x_value_to_pixel_aggregate, m_y_value_to_pixel_aggregate,
				m_aggregate_width, m_aggregate_height, m_aggregate_image_pitch,
				m_x_min, m_x_max, m_y_min, m_y_max,
				m_trace_color[i], m_margin, mp_d_chart_selected);
	}

	// copy full aggregate image to host
	cudaMemcpy2D(mp_h_aggregate_image, m_cols * sizeof(uchar4), mp_d_aggregate_image, m_aggregate_image_pitch, 
		m_cols * sizeof(uchar4), m_rows, cudaMemcpyDeviceToHost);

}



uchar4 * CudaChartArray::GetAggregateImagePtr()
{
	cudaError_t result = cudaMemcpy2D(mp_h_aggregate_image, m_aggregate_width * sizeof(uchar4), 
		mp_d_aggregate_image, m_aggregate_image_pitch, 
		m_aggregate_width * sizeof(uchar4), m_aggregate_height, cudaMemcpyDeviceToHost);

	return mp_h_aggregate_image;
}

void CudaChartArray::Resize(int chartArrayWidth, int chartArrayHeight, int aggregateWidth, int aggregateHeight)
{
	// clean up and resize chart image
	m_chartArray_width = chartArrayWidth;
	m_chartArray_height = chartArrayHeight;

	m_chart_width = (chartArrayWidth - (2 * m_margin) - ((m_cols - 1)*m_padding)) / m_cols;
	m_chart_height = (chartArrayHeight - (2 * m_margin) - ((m_rows - 1)*m_padding)) / m_rows;

	cudaError_t result = cudaFree(mp_d_chart_image);
	bool success = AllocateForChartImage();
	if (!success)
	{
		success = true;	
	}
	result = cudaFree(mp_h_chart_image);
	result = cudaMallocHost((uchar4**)&mp_h_chart_image, m_chartArray_width * m_chartArray_height * sizeof(uchar4));
	if (result != cudaSuccess)
	{
		result = cudaSuccess;
	}
	
	// clean up and resize aggregate image
	m_aggregate_width = aggregateWidth;
	m_aggregate_height = aggregateHeight;
	result = cudaFree(mp_d_aggregate_image);
	success = AllocateForAggregateImage();
	if (!success)
	{
		success = true;
	}
	result = cudaFree(mp_h_aggregate_image);	
	result = cudaMallocHost((uchar4**)&mp_h_aggregate_image, m_aggregate_width * m_aggregate_height * sizeof(uchar4));
	if (result != cudaSuccess)
	{
		result = cudaSuccess;
	}


	CalcConversionFactors();

	Redraw();

	RedrawAggregate();	
}



void CudaChartArray::Reset()
{
	for (int i = 0; i < m_numTraces; i++)
	{
		m_num_data_points[i] = 0;
	}

	m_x_min = m_initial_xRange.x;
	m_x_max = m_initial_xRange.y;
	m_y_min = m_initial_yRange.x;
	m_y_max = m_initial_yRange.y;
}



bool* CudaChartArray::GetSelectionArrayPtr()
{
	return mp_h_chart_selected;
}
