import cv2
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
##           Método para generar circulo en el dominio de fourier           ##
##############################################################################  
def getFilters(r,width,height):
    w_ =width/2
    h_ = height/2
    x=np.linspace(-w_, w_-1,width)
    y = np.linspace(-h_, h_-1,height)
    [x,y] = np.meshgrid(x,y)
    z = np.sqrt(x**2+y**2)
    cL = z < r
    cH = ~cL
    return (cL, cH)
##############################################################################
##            Método para Obtener el contraste de la imagen                 ##
##############################################################################  
def getContrast(img):
    norm_image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return norm_image.std()

def algoritmo(imgsrc):
        img = imgsrc[:,:,1]
        height, width =np.shape(img)
        ##############################################################
        ##             Threshold mácula y disco óptico              ##
        ##############################################################
        px_total = height*width
        th_disco = px_total*0.0015
        th_macula = px_total*0.005
        th_vasos = px_total*0.03
        contrast = getContrast(img)
        ##############################################################
        ##                    Aplicación de filtro                  ##
        ##############################################################
        ft = np.fft.fftshift(np.fft.fft2(img))
        #   Filtro Low Pass
        [cL, cH] = getFilters(10,width,height)
        l_ft = ft * cL 
        #   Filtro Band pass
        [cL1, cH1] = getFilters(20,width,height)
        [cL2, cH2] = getFilters(150,width,height)
        cBP = cH1 * cL2
        cBP = cBP.astype(float)
        cBP = cv2.GaussianBlur(cBP, ksize=(0,0),sigmaX=3,borderType=cv2.BORDER_REPLICATE)    
        h_ft = ft * cBP
        #####              Reconstrucción de la imagen              ##
        low_filtered_image = np.fft.ifft2(np.fft.ifftshift(l_ft))
        high_filtered_image = np.fft.ifft2(np.fft.ifftshift(h_ft))
        
        low_f = abs(abs(low_filtered_image))
        high_f = abs(abs(low_filtered_image))
        ##############################################################
        ###                Búsqueda del disco óptico               ###
        ##############################################################
        D1 = 40
        A= np.zeros(np.shape(img))
        endx = np.shape(A)[1]-1
        endy = np.shape(A)[0]-1
        
        std_high = np.std(np.reshape(np.real(high_filtered_image),width*height))    
        A[np.where(-np.real(high_filtered_image)>std_high)] = 1
        A[1:D1, :] = 0
        A[:, 1:D1] = 0   
        A[:,endx-D1:endx] = 0
        A[endy-D1:endy,:] = 0
        suma_hp = np.sum(A)   
        ##################################################################
        ###     Sumatoria vertical para la búsqueda del disco óptico    ##
        ##################################################################
        num  = 12
        C = np.zeros([num,1])
        nx = np.floor(height/num).astype(int)
        ny = np.floor(width/num).astype(int)
        
        for col in range(0,num):
            C[col] = np.sum(A[50:endx-50, col* ny+1: (col+1)*ny])
            
        x = np.arange(0, num,1)*nx+nx/2
        y = np.arange(0, num,1)*ny+ny/2
        #   Búsqueda de la posición del disco óptico   
        b =np.where(C == np.amax(C))[0]
        
        #   Búsqueda del máximo en low pass
        img_disco = np.zeros(np.shape(img))
        ind_disco= np.arange(round(y[max(b[0]-1,1)]),round(y[min(b[0]+1,12)]),1)
        
        thr_disco = 0.95*np.max(low_f[D1:endy-D1,ind_disco])
        img_disco[:,ind_disco] = low_f[:,ind_disco] >thr_disco
        ##############################################################
        ##         Búsqueda de la posición central del disco        ##
        ##############################################################
        ind = np.where(img_disco == 1)
        indx_disco = ind[1]
        indy_disco = ind[0]  
        pos_discox = np.floor(np.median(indx_disco))
        pos_discoy = np.floor(np.median(indy_disco))
        ##############################################################
        ##                 Búsqueda de la mácula                    ##
        ##############################################################    
        img_macula = np.zeros(np.shape(img))
        D=60
        ind = np.arange(max(pos_discoy-90, D),min(pos_discoy+90,np.shape(img)[0]-D),1).astype(int)
        cuad_x = np.floor(np.shape(img)[1]/2)
        cuad_y = np.floor(np.shape(img)[0]/5)

        if(pos_discox > np.shape(img)[1]/2):
            ind_x = np.arange(max(pos_discox-cuad_x,D),min(pos_discox-cuad_y,np.shape(img)[1]-D),1).astype(int)
        else:
            ind_x = np.arange(max(pos_discox+cuad_y,D),min(pos_discox+cuad_x,np.shape(img)[1]-D),1).astype(int)

        thr_macula = np.min(low_f[np.ix_(ind,ind_x)])*1.1
        img_macula[np.ix_(ind,ind_x)] = low_f[np.ix_(ind,ind_x)]<thr_macula

        size_macula = np.sum(img_macula)
        size_disco = np.sum(img_disco)
        if(suma_hp > th_vasos*0.5 and suma_hp<th_vasos*2.5):
            vessels_detected = 1
            if(size_disco > th_disco*0.25 and size_disco < th_disco*1.85):
                disco_detected = 1
                if(size_macula > th_macula*0.15 and size_macula < th_macula*2):
                    macula_detected = 1
                else:
                    macula_detected = 0
            else:
                disco_detected = 0
                macula_detected = 0
        else:
            vessels_detected = 0
            disco_detected = 0
            macula_detected = 0
        ##############################################################
        ##                 Generación de gráficos                   ##
        ##############################################################
        linea1 = [round(y[max(b[0]-1,1)]),round(y[max(b[0]-1,1)])]
        linea2 = [round(y[min(b[0]+1,12)]),round(y[min(b[0]+1,12)])]
        #   Gráfico de la imagen original   
        plt.subplot(221)
        plt.title("Vasos sanguineos: {}, Disco: {}, Mácula: {} ".format(vessels_detected,disco_detected,macula_detected))
        plt.imshow(imgsrc)
        plt.contour(img_disco,colors="red")
        plt.contour(img_macula, colors= "cyan")
        #Área de busqueda disco
        plt.plot(linea1,[1, height],color="y")
        plt.plot(linea2,[1, height],color="y")        
        #Área de busqueda macula
        plt.plot([ind_x[0],ind_x[len(ind_x)-1]],[ind[0],ind[0]], color="g")
        plt.plot([ind_x[0],ind_x[len(ind_x)-1]],[ind[len(ind)-1],ind[len(ind)-1]], color="g")
        plt.plot([ind_x[len(ind_x)-1], ind_x[len(ind_x)-1]],[ind[0], ind[len(ind)-1]], color="g")
        plt.plot([ind_x[0], ind_x[0]],[ind[len(ind)-1], ind[0]], color="g")
        #   Gráfico suma vertical
        plt.subplot(222)
        C= np.reshape(C,[12,])
        plt.bar(y, C,width=50, color="b")
        #   Área de búsqueda disco
        plt.plot(linea1,[1, np.max(C)],color="r")
        plt.plot(linea2,[1, np.max(C)],color="r")
        #   Gráfico de la imagen filtrada en low pass
        plt.subplot(223)    
        plt.imshow(abs(low_filtered_image))
        plt.contour(img_disco,colors="red")
        plt.contour(img_macula, colors= "cyan")
        #   Área de búsqueda disco
        plt.plot(linea1,[1, height],color="y")
        plt.plot(linea2,[1, height],color="y")
        #   Cuadro de búsqueda
        plt.plot([ind_x[0],ind_x[len(ind_x)-1]],[ind[0],ind[0]], color="g")
        plt.plot([ind_x[0],ind_x[len(ind_x)-1]],[ind[len(ind)-1],ind[len(ind)-1]],color="g")
        plt.plot([ind_x[len(ind_x)-1], ind_x[len(ind_x)-1]],[ind[0], ind[len(ind)-1]],color="g")
        plt.plot([ind_x[0], ind_x[0]],[ind[len(ind)-1], ind[0]],color="g")
        #   Gráfico de la imagen filtrada en high pass
        plt.subplot(224)
        plt.imshow(abs(A))
        plt.plot(linea1,[1, height],color="y")
        plt.plot(linea2,[1, height],color="y")
        plt.show()
        return vessels_detected, disco_detected, macula_detected

def menu():
    while(1):
        opcion = ""
        while (not opcion.isdigit()):
            print("Seleccione el dataset a utilizar: ")
            print("1. DRIMDB")
            print("2. DRIVE")
            print("3. Salir")
            opcion = input("Ingrese una opción: ")
        opcion = int(opcion)
        if(opcion == 1):
            main(dataset="DRIMDB")
        if(opcion == 2):
            main(dataset="DRIVE")
        if(opcion == 3):
            break


    

def main(dataset="DRIMDB"):
    deteccion_vasos = list()
    deteccion_disco = list()
    deteccion_macula = list()
    if(dataset == "DRIMDB"):
        for i in range(1,125):
            imgsrc = cv2.imread("DRIMDB/Good/drimdb_good ({}).jpg".format(i))
            vasos,disco,macula = algoritmo(imgsrc)
            deteccion_vasos.append(vasos)
            deteccion_disco.append(disco)
            deteccion_macula.append(macula)

    if(dataset == "DRIVE"):
        for i in range(1,40):
            if(i<=20):
                if(i<10):
                    imgsrc = cv2.imread("DRIVE/0{}_test.tif".format(i))
                else:
                    imgsrc = cv2.imread("DRIVE/{}_test.tif".format(i))
            
            if(i>20 and i<=40):
                imgsrc = cv2.imread("DRIVE/{}_training.tif".format(i))
            vasos,disco,macula = algoritmo(imgsrc)
            deteccion_vasos.append(vasos)
            deteccion_disco.append(disco)
            deteccion_macula.append(macula)
    print("Vasos sanguineos: {}".format(np.sum(deteccion_vasos)))
    print("Disco: {}".format(np.sum(deteccion_disco)))
    print("Macula: {}".format(np.sum(deteccion_macula)))
    print("Total: {}".format(np.sum(deteccion_vasos)+np.sum(deteccion_disco)+np.sum(deteccion_macula)))
menu()