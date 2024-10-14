# 速通FreeRTOS
## 环境搭建：基于HAL库，keil开发
由于个人技术力不足，无法实现平台移植：有关CMSIS了解的太少了，去找开发板对应的模板吧（
1. 下载hal库：用cubemx或者github都行  
    * 添加MDK_ARM: Drivers\CMSIS\Device\ST\STM32F1xx\Source\Templates\arm
    * 添加库文件：Drivers\STM32F1xx_HAL_Driver\src 将需要的文件添加进工程
    * 添加编译路径： Drivers\STM32F1xx_HAL_Driver\Inc；Drivers\STM32F1xx_HAL_Driver\Inc\Legacy 
1. 下载CMSIS：Github下载，下载source code而不是安装包，否则需要在包管理器折腾
    * 添加编译路径：CMSIS/Device/ST/STM32F1xx/Include；CMSIS/Include； 
1. 导入FreeRTOS：Github下载  
    * 添加库文件：FreeRTOSv202212.01\FreeRTOS\Source 中的.c文件加入工程，这些就是kernel
    * 添加库文件：Source\portable\RVDS，找到使用的mpu内核；Source\portable\MemMang，文件夹中选择合适的内存管理文件
    * 添加编译路径： include；
    * 新建FreeRTOSConfig.h：有关FreeRTOS的配置，可以剪枝不需要的功能节省ram

## 什么是RTOS
RTOS，Real Time Operating System，实时操作系统，一类针对微控制器的操作系统而非单独的系统，不同于裸机开发   
初学者直观的一个东西是他可以同时执行多个函数，通过在任务间快速切换实现，提供了对任务（函数包装）的管理：运行，就绪，阻塞，挂起
> QMK键盘固件在arm上基于chibios实现，该RTOS主攻轻量化，ram可低至1kb  
> 如果你闲的蛋疼也可以自己实现一个，有些芯片的协议栈本身就是一个rtos  
对于一些flash和ram稀缺的mcu，或者有着严格时序要求的功能，rtos就不太适合  
工程量偏大，功能耦合的项目则更适合rtos，有效降低屎山

## 回忆HAL库
1. GPIO操控  
    本质：操作寄存器
    ```c
    HAL_GPIO_WritePin(GPIOC, GPIO_PIN_13, GPIO_PIN_SET);
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_10, GPIO_PIN_RESET);
    ```
1. 中断  
    本质：还是寄存器：EXIT(External interupt/event controller),NVIC(Nested Vectored interupt controller)
    ```c
    GPIO_InitTypeDef GPIO_InitStruct；
    GPIO_InitStruct.Pin = GPIO_PIN_14;                  // 哪个针脚触发
    GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING_FALLING; // 触发条件：上升沿，下降沿，两个
    GPIO_InitStruct.Pull = GPIO_NOPULL;                 // 是否上下拉
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

    // 配置NVIC
    HAL_NVIC_SetPriority(EXTI15_10_IRQn, 0, 0);         // 配置优先级
    HAL_NVIC_EnableIRQ(EXTI15_10_IRQn);                 // 使能
    ```
    * 中断优先级：抢占优先级，响应优先级

## 快速上手：多线程与线程交互
### 任务：多线程
1. 基本概念
    1. 在FreeRTOS中任务无非四种状态：**运行，就绪，阻塞，挂起**  
    通过快速的任务间切换（低至us级别）来实现多个线程同时进行，
        * Running运行：正在执行
            * 只能由就绪态产生
        * Ready就绪：随时可以执行
            > 对于相同优先级的任务取决于调度策略：抢占，轮流来/干等。  
            > 即使开启任务优先级抢占，中断始终先于任务优先级  
            > 可由FreeRTOSConfig.h配置更多细节
        * Blocked阻塞：等待进一步通知
            * 只能在运行态执行阻塞api
        * Suspended挂起：暂时用不着
            * 随时能挂到路灯上
    1. 具体函数：
        > 所有函数均操作的是任务句柄handle
        > FreeRTOSConfig.h配置：  
        > INCLUDE_vTaskSuspend? 启用挂起任务:不启用  
        > INCLUDE_xTaskResumeFromISR? 启用中断挂起:不启用
        >
        >
        >
        ```c
        void vTaskStartScheduler(): 启动调度
        void vTaskStartScheduler(): 结束调度
        void vTaskSuspend(): 挂起
        void vTaskResume(): 恢复被挂起的任务
        void xTaskResumeFromISR(): 在中断时恢复被挂起的内容
        void vTaskDelete(): 删除任务
        ```
1. 任务间：同步，互斥与通信    
一般来说，函数中不使用静态变量实现互斥操作：多个任务可以同时调用一个函数，或者在执行中被挂起/中断   
一种不负责任的解决办法：操作时关闭中断
1. 创建：
    > FreeRTSOConfig.h配置：   
    > configSUPPORT_DYNAMIC_ALLOCATION ? 启用xTaskCreate():不启用 // 动态内存管理也由该项启动  
    > configSUPPORT_STATIC_ALLOCATION ? 启用xTaskCreateStatic():不启用  
    > configSUPPORT_STATIC_ALLOCATION；portUSING_MPU_WRAPPERS ? 启用xTaskCreateRestrictedStatic:不启用
    > INCLUDE_vTaskDelete? 启用vTaskDelete():不启用
    >
    >
    >
    >
    ```c
    /*动态创建任务*/
    BaseType_t xTaskCreate ( 
        TaskFunction_t pxTaskCode,                  // 函数指针, 任务函数
        const char * const pcName,                  // 任务的名字
        const configSTACK_DEPTH_TYPE usStackDepth,  // 栈大小,单位为word,10表示40字节,一般就是猜的
        void * const pvParameters,                  // 调用任务函数时传入的参数
        UBaseType_t uxPriority,                     // 优先级
        TaskHandle_t * const pxCreatedTask          // 任务句柄, 以后使用它来操作这个任务
    ); 

    /*静态创建任务*/
    TaskHandle_t xTaskCreateStatic ( 
        TaskFunction_t pxTaskCode,                  // 函数指针, 任务函数
        const char * const pcName,                  // 任务的名字
        const uint32_t ulStackDepth,                // 栈大小,单位为word,10表示40字节
        void * const pvParameters,                  // 调用任务函数时传入的参数
        UBaseType_t uxPriority,                     // 优先级
        StackType_t * const puxStackBuffer,         // 静态分配的栈，就是一个buffer
        StaticTask_t * const pxTaskBuffer           // 静态分配任务结构体指针，用它来操作这个任务
    );
    // 上面两种函数返回值为pdPASS时，表示创建成功
    /* 静态创建受MPU保护的任务*/
    BaseType_t xTaskCreateRestrictedStatic(
        const TaskParameters_t * const pxTaskDefinition, // 指向任务参数结构体的指针
        TaskHandle_t * pxCreatedTask
    );
    // 创建成功返回句柄
    ```
1. 删除
    ```c
    void vTaskDelete(TaskHandle_t xTaskToDelete);
    ```


1. 延迟：
    ```c
    void vTaskDelay(const TickType_t xTicksToDelay) // 对systick计数，计数达到n时退出  
    ```
    * 注意：如果一个任务在tick间完成那么延迟会略高于n tick
    ```c 
    vTaskDelayUntil(TickType_t * const pxPreviousWakeTime
                    const TickType_t xTimeIncrement
                   ) 
    ```
    * 该任务与延迟之和的周期为n tick ~~没说运行时间太长了会怎样~~

### 列表：双链表实现
```c
typedef struct xLIST {
    listFIRST_LIST_INTEGRITY_CHECK_VALUE                    // 校验值
    volatile UBaseType_t    uxNumberOfItems;                // 列表中列表项的数量
    ListItem_t *            configLIST_VOLATILE pxIndex;    // 某一个列表项，用于遍历列表
    MiniListItem_t          xListEnd;                       // 最后一个列表项
    listSECOND_LIST_INTEGRITY_CHECK_VALUE                   // 校验值 
} List_t;
```
* 函数项：~~为什么要有校验项~~   
分为一般与迷你，迷你缺少列表拥有者、所在列表、第一个校验项
* 函数操作
    ```c
        void vListInitialise()          // 初始化
        void vListInitialiseItem()      // 初始化列表项
        void vListInsertEnd()           // 列表末尾插入列表项
        void vListInsert()              // 插入列表项
        void uxListRemove()             // 移除列表项
        // 更多的宏...
    ```
### 线程交互
#### 队列：在任务与中断间传递信息
* 应该不需要复习数据结构吧：一边进，一边出  ~~首先，咱们先定义下什么是头什么是尾~~
* 如果队列满/空，有任务写/读该队列则会进入阻塞态，同优先级下等待时间最久优先  
> 有关FreeRTOS_config.h  
> portMAX_DELAY:最长阻塞时间，单位tick，如果被设为该值则永远不会超时返回
1.  创建
    ```c
    // 创建列表
    QueueHandle_t xQueueCreate(
        UBaseType_t uxQueueLength,   // 队列长度
        UBaseType_t uxItemSize       // 数据大小
    );
    // 静态创建列表
    QueueHandle_t xQueueCreateStatic(
        UBaseType_t uxQueueLength,      // 同上
        UBaseType_t uxItemSize,         // 同上
        uint8_t     *pucQueueStorageBuffer, // 缓冲区：如果uxQUeueLength非0，则为上面两个乘积
        StaticQueue_t *pxQueueBuffer    // 缓冲区：保存数据结构
    );
    // 复位
    BaseType_t xQueueReset(QueueHandle_t pxQueue);
    // 删除
    void vQueueDelete(QueueHandle_t xQueue);
    ```
1. 读写操作
    ```c
    // 写队列尾
    BaseType_t xQueueSend( // 同xQueueSendToBack()
        QueueHandle_t xQueue,       // 指定的队列句柄
        const void *pvItemToQueue,  // 要写入数据的指针，创建队列时已经指定了数据大小
        TickType_t xTicksToWait     // 阻塞时间
    );  // 之后的重复参数不再阐述

    // ISR(Interupt Service Routine)写队列尾 中断使用 不可阻塞
    BaseType_t xQueueSendToBackFromISR (
        QueueHandle_t xQueue,
        const void *pvItemToQueue,
        BaseType_t *pxHigherPriorityTaskWoken 
        // 用于唤醒优先级更高的任务，可以是NULL以禁用，之后不再赘述
    );
    // 写队列头
    BaseType_t xQueueSendToFront (
        QueueHandle_t xQueue,
        const void *pvItemToQueue,
        TickType_t xTicksToWait
    );
    // 中断写队列头
    BaseType_t xQueueSendToFrontFromISR (
        QueueHandle_t xQueue,
        const void *pvItemToQueue,
        BaseType_t *pxHigherPriorityTaskWoken
    );
    /* 读队列头，被读取的数据出队列 */
    // 读队列头
    BaseType_t xQueueReceive (QueueHandle_t xQueue,
        void * const pvBuffer,
        TickType_t xTicksToWait // 阻塞最大时间
    );
    // 中断读队列头
    BaseType_t xQueueReceiveFromISR (
        QueueHandle_t xQueue,
        void *pvBuffer,
        BaseType_t *pxTaskWoken
    );
    ```
1. #查询队列状态#
    ```c
    // 可用数据个数
    UBaseType_t uxQueueMessagesWaiting (const QueueHandle_t xQueue);
    // 可用空间
    UBaseType_t uxQueueSpacesAvailable (const QueueHandle_t xQueue);
    ```
#### 信号量：状态机
可通知(二进制信号量，限制为1),可计数(计数信号量)
1. 创建
    ```c
    SemaphoreHandle_t xSemaphoreCreateBinary(void) // 动态创建一个通知信号量，返回句柄
    SemaphoreHandle_t xSemaphoreCreateBinaryStatic(StaticSemaphore_t *pxSemaphoreBuffer); // 静态创建，需要先有一个StaticSemaphore_t的结构体
    SemaphoreHandle_t xSemaphoreCreateCounting( // 动态 计数信号量
        UBaseType_t uxMaxCount,                 // 最大值
        UBaseType_t uxInitialCount              // 初始值
    )
    ```
1. 删除
    ```c
    void vSemaphoreDelete (SemaphoreHandle_t xSemaphore);
    ```
1. 操作
    ```c
    // 两种信号量的操作函数一致
    // 发货，xSEmaphore为被操作信号量的句柄
    BaseType_t xSemaphoreGive (SemaphoreHandle_t xSemaphore);
    // ISR版本
    BaseType_t xSemaphoreGiveFromISR (
        SemaphoreHandle_t xSemaphore,
        BaseType_t *pxHigherPriorityTaskWoken
    );
    // 取货
    BaseType_t xSemaphoreTake(
        SemaphoreHandle_t xSemaphore,
        TickType_t xTicksToWait
    );
    BaseType_t xSemaphoreTakeFromISR(
        SemaphoreHandle_t xSemaphore,
        BaseType_t *pxHigherPriorityTaskWoken
    );
    ```
1. 一些细节：
    * 通知信号量在Give后的Give会丢失，如要传递额外信息请建立缓冲区，通知后写入，获取时一起获取

#### 互斥量
阻止多任务使用相同资源导致问题：使用全局变量，静态变量，外设。  
解决办法：茅厕占坑，把自己锁上
> FreeRTOSConfig选项:  
> #define configUSE_MUTEXES ? 启用互斥量:不启用
* 可通过信号量，队列实现类似的功能
* 可能会出现优先级反转：低优先级的任务占坑，高优先级任务阻塞；
    * 互斥量带有优先级继承以改善：高优先级任务的优先级暂时借给低优先级(take)用，直到释放互斥锁(give)
1. 创建
    ```c
    SemaphoreHandle_t xSemaphoreCreateMutex (void);
    SemaphoreHandle_t xSemaphoreCreateMutexStatic (StaticSemaphore_t *pxMutexBuffer);
    ```
1. 操作：同信号量，但注意互斥量不能在ISR中使用，但函数操作可以
* 死锁：相互有对方的锁但是都处于阻塞态/两次调用同一个锁
    * 解决办法：递归锁,仅函数可以take多次，但必须give相同次数才被释放
    ```c
    // 创建，成功创建返回句柄
    SemaphoreHandle_t xSemaphoreCreateRecursiveMutex()
    // 获得
    SemaphoreHandle_t xSemaphoreTakeRecursive (SemaphoreHandle_t xSemaphore)
    // 释放
    SemaphoreHandle_t xSemaphoreGiveRecursive (
        SemaphoreHandle_t xSemaphore,
        TickType_t xTicksToWait
    )
    ```



#### 任务通知
* 更节省内存，效率更高
* 不能给中断任务传数据；独享通知与数据；无buffer；通知不能被阻塞；
1. 通知状态：
```c
typedef struct tskTaskControlBlock {
 ......
    /* configTASK_NOTIFICATION_ARRAY_ENTRIES = 1 */
    volatile uint32_t ulNotifiedValue[ configTASK_NOTIFICATION_ARRAY_ENTRIES ]; // 通知值
    volatile uint8_t ucNotifyState[ configTASK_NOTIFICATION_ARRAY_ENTRIES ];    // 通知状态
} tskTCB;

```
* 三种通知状态：前两个待确认
    1. taskNOT_WAITING_NOTIFICATION：等待通知 
    1. taskWAITING_NOTIFICATION：没有等待通知
    1. taskNOTIFICATION_RECEIVED：接受到通知，待处理
1. 通知函数：简化版/专业版
```c
// 发出通知，简化版，均使得通知值++，通知状态为pending
BaseType_t xTaskNotifyGive (TaskHandle_t xTaskToNotify);
void vTaskNotifyGiveFromISR (
    TaskHandle_t xTaskHandle, 
    BaseType_t *pxHigherPriorityTaskWoken // 被通知任务的句柄，会变为就绪态。如果唤醒的是更高优先级的任务，则设置为pdTRUE
)

// 取出通知，简化版
uint32_t ulTaskNotifyTake( 
    BaseType_t xClearCountOnExit,       // 函数返回前是清零还是-1? pdTRUE:pdFALSE
    TickType_t xTicksToWait             // 超时时间
)
// 通知值为0进入阻塞态
```

### 事件组
不同于队列，信号量：可以唤醒多个任务，保持当前状态
> FreeRTOSConfig.h配置  
> configUSE_16_BIT_TICKS ? 低8位事件表示:低24位事件表示 // 由MCU决定
1. 创建
```c
EventGroupHandle_t xEventGroupCreate (void);
EventGroupHandle_t xEventGroupCreateStatic (StaticEventGroup_t *pxEventGroupBuffer);
```
1. 删除
```c
void vEventGroupDelete (EventGroupHandle_t xEventGroup)
```
1. 操作
```c
EventBits_t xEventGroupSetBits (        // 任务中使用
    EventGroupHandle_t xEventGroup,     // 被操作的事件组
    const EventBits_t uxBitsToSet       // 被设置的位
); // 返回原事件值（没啥用）

BaseType_t xEventGroupSetBitsFromISR (  // 中断中使用
    EventGroupHandle_t xEventGroup,
    const EventBits_t uxBitsToSet,
    BaseType_t *pxHigherPriorityTaskWoken // 是否使得更高优先级任务就绪 ? pdTRUE:pdFALSE
);

EventBits_t xEventGroupWaitBits(        // 等待事件
    EventGroupHandle_t xEventGroup,     // 等待哪个事件组
    const EventBits_t uxBitsToWaitFor,  // 等待哪些位, eg:0110表示等待bit1和bit2
    const BaseType_t xClearOnExit,      // 退出前是否清除位? pdTRUE:pdFALSE
    const BaseType_t xWaitForAllBits,   // 条件是否是与门? pdTRUE:pdFALSE
    TickType_t xTicksToWait             // 超时时间
    // 0：即刻返回，portMAX_DELAY：成功才返回，期望值
);

EventBits_t xEventGroupSync(            // 同步任务：表明自己完成任务，等待别人
    EventGroupHandle_t xEventGroup,     // 等待哪个事件组
    const EventBits_t uxBitsToSet,      // 设置那些事件
    const EventBits_t uxBitsToWaitFor,  // 等待哪些位
    TickType_t xTicksToWait             // 超时时间
);
```